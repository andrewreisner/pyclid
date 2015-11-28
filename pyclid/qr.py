import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.reduction import ReductionKernel
import pyopencl_blas as blas

import pyclid.util as util


def iddr_qrpiv(queue, m, n, a, krank, ind, ss):
    r = cl_array.Array(queue, a.shape, a.dtype)

    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    qr_prg = cl.Program(ctx, util.get_source('qr_kerns.cl')).build()

    # begin debug
    m = 6
    n = 3
    adebug = np.zeros((m,n), dtype=a.dtype)
    adebug[:,0] = np.array([1,0,0,-1,-1,0])
    adebug[:,1] = np.array([0,1,0,1,0,-1])
    adebug[:,2] = np.array([0,0,1,0,1,1])

    cla = cl_array.Array(queue, adebug.shape, adebug.dtype)
    cla.set(adebug)
    ssdebug = cl_array.Array(queue, n, dtype=ss.dtype)
    rdebug = cl_array.Array(queue, cla.shape, cla.dtype)
    rdebug.set(np.zeros(cla.shape))
    # for i in range(n):
    #     print(np.sum(adebug[:,i]**2))
    # end debug

    ss_l = cl.LocalMemory(m*np.dtype('float64').itemsize)
    # qr_prg.ss(queue, [m,n], [m,1], a.data, ss.data, ss_l)
    # kpiv = util.argmax(queue, ss)
    qr_prg.ss(queue, [m,n], [m,1], cla.data, ssdebug.data, ss_l)

    qk = cl.LocalMemory(m*np.dtype('float64').itemsize)
    aj_qk = cl.LocalMemory(m*np.dtype('float64').itemsize)

    nloops = np.min([m,n,krank])
    for k in range(nloops-1):
        # ind[k] = kpiv
        # qr_prg.swap_col(queue, [m, 1], [m, 1], a.data,
        #                 np.int32(k), np.int32(kpiv), np.int32(n))
        # ss[k], ss[kpiv] = ss[kpiv], ss[k]
        # qr_prg.proj_rm(queue, [m, n-k], [m, 1],
        #                a.data, r.data, ss.data,
        #                qk, ss_l, aj_qk, k, n)
        qr_prg.proj_rm(queue, [m, n-(k+1)], [m, 1],
                       cla.data, rdebug.data, ssdebug.data,
                       qk, ss_l, aj_qk, np.int32(k), np.int32(n))
        #print(cla.get())
        qr_prg.ss(queue, [m,n], [m,1], cla.data, ssdebug.data, ss_l)
        #print(rdebug.get())
        #print(ssdebug.get())
    qr_prg.norm(queue, [m,1], [m, 1],
                rdebug.data, ssdebug.data,
                np.int32(nloops-1), np.int32(n))
    print(rdebug.get())
    #print(rdebug.get())
