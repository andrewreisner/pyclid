import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.reduction import ReductionKernel
import pyopencl_blas as blas

import pyclid.util as util


def iddr_qrpiv(queue, m, n, a, krank, ind, ss):
    r = cl_array.Array(queue, a.shape, a.dtype)
    ind.set(np.arange(ind.shape[0], dtype=ind.dtype))

    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    qr_prg = cl.Program(ctx, util.get_source('qr_kerns.cl')).build()

    # begin debug
    m = 7
    n = 4
    # anp = np.zeros((m,n), dtype=a.dtype)
    # anp[:,0] = np.array([1,0,0,-1,-1,0])
    # anp[:,1] = np.array([0,1,0,1,0,-1])
    # anp[:,2] = np.array([0,0,1,0,1,1])
    from scipy import random
    anp = random.randn(m,n)

    a = cl_array.Array(queue, anp.shape, anp.dtype)
    a.set(anp)
    ss = cl_array.Array(queue, n, dtype=a.dtype)
    r = cl_array.Array(queue, a.shape, a.dtype)
    r.set(np.zeros(a.shape))
    ind = cl_array.Array(queue, n, dtype=np.int32)
    ind.set(np.arange(n, dtype=ind.dtype))

    from scipy import linalg
    _,R,P = linalg.qr(anp,pivoting=True)
    print(R)
    print(P)
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
        kpiv = util.argmax(queue, ss[k+1:]) + k + 1

    qr_prg.norm(queue, [m,1], [m, 1],
                r.data, ss.data,
                np.int32(nloops-1), np.int32(n))
    # only really need up to krank here
    print(r.get())
    print(ind)
