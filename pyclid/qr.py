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
    # m = 4
    # n = 3
    # adebug = np.zeros((m,n), dtype=a.dtype)
    # adebug[:,0] = np.linspace(0,1,m)
    # adebug[:,1] = np.linspace(1,2,m)
    # adebug[:,2] = np.linspace(2,3,m)

    # cla = cl_array.Array(queue, adebug.shape, adebug.dtype)
    # cla.set(adebug)
    # ssdebug = cl_array.Array(queue, n, dtype=ss.dtype)
    # for i in range(n):
    #     print(np.sum(adebug[:,i]**2))
    # end debug

    ss_l = cl.LocalMemory(m*np.dtype('float64').itemsize)
    qr_prg.ss(queue, [m,n], [m,1], a.data, ss.data, ss_l)
    kpiv = util.argmax(queue, ss)

    nloops = np.min([m,n,krank])
    for k in range(nloops):
        ind[k] = kpiv
        qr_prg.swap_col(queue, [m, 1], [m, 1], a.data,
                        np.int32(k), np.int32(kpiv), np.int32(n))
        ss[k], ss[kpiv] = ss[kpiv], ss[k]
        qr_prg.proj_rm(queue, [m, n-k], [m, 1], a.data,
                       ss_l)
