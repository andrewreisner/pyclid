import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.reduction import ReductionKernel
import pyopencl_blas as blas

import pyclid.util as util


def iddr_qrpiv(queue, m, n, a, krank, ind, ss):
    r = cl_array.Array(queue, a.shape, a.dtype)
    r.set(np.zeros(a.shape, a.dtype))
    ind.set(np.arange(ind.shape[0], dtype=ind.dtype))

    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    qr_prg = cl.Program(ctx, util.get_source('qr_kerns.cl')).build()

    # begin debug
    # from scipy import linalg
    # _,R,P = linalg.qr(a.get(),pivoting=True)
    # print(R)
    # print(P)
    # end debug

    ss_l = cl.LocalMemory(m*np.dtype('float64').itemsize)
    qr_prg.ss(queue, [m,n], [m,1], a.data, ss.data, ss_l,
              np.int32(0),np.int32(n))
    kpiv = util.argmax(queue, ss)

    qk = cl.LocalMemory(m*np.dtype('float64').itemsize)
    aj_qk = cl.LocalMemory(m*np.dtype('float64').itemsize)

    nloops = np.min([m,n,krank])
    for k in range(nloops-1):
        qr_prg.swap_col(queue, [m, 1], [m, 1], a.data, r.data, ss.data, ind.data,
                        np.int32(k), np.int32(kpiv), np.int32(n))
        qr_prg.proj_rm(queue, [m, n-(k+1)], [m, 1],
                       a.data, r.data, ss.data,
                       qk, ss_l, aj_qk, np.int32(k), np.int32(n))
        qr_prg.ss(queue, [m,n-(k+1)], [m,1], a.data, ss.data, ss_l, np.int32(k+1),
                  np.int32(n))
        queue.finish()
        kpiv = util.argmax(queue, ss[k+1:]) + k + 1

    qr_prg.norm(queue, [m,1], [m, 1],
                r.data, ss.data,
                np.int32(nloops-1), np.int32(n))
    # print(r.get())
    # print(ind)
    # copy r into a for output
    a[:krank,:] = r[:krank,:]
