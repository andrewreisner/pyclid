#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator

import pyopencl as cl
from pyopencl.array import Array
import pyopencl_blas as blas

import pyclid

def setup_lowrank(n, dtype='float64'):
    A0 = np.zeros((n,n), dtype=dtype)
    A0[...] = np.random.randn(*A0.shape)
    U0, _, VT0 = la.svd(A0)
    sigma = np.exp(-np.arange(n))
    A = (U0 * sigma).dot(VT0)

    return A


def main():
    ctx = cl.create_some_context()
    # devices = ctx.get_info(cl.context_info.DEVICES)
    # print(devices[0].get_info(cl.device_info.VERSION))
    queue = cl.CommandQueue(ctx, properties=0)

    dtype = 'float64'
    n = 500
    k = 30

    A = setup_lowrank(n, dtype=dtype)
    #mvt = pyclid.util.setup_matvect(queue, A)

    print('finished setup')
    L = pyclid.util.setup_op(queue, A)
    idx, proj = pyclid.interp_decomp(queue, L, k)
    #idx, proj = pyclid.iddr_rid(queue, n, n, mvt, k)

    # begin debug
    import scipy.linalg as la
    import scipy.linalg.interpolative as sli
    from scipy.sparse.linalg import aslinearoperator
    B = A[:,idx[:k]]
    P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
    Aapprox = np.dot(B,P)
    print(la.norm(A - Aapprox, 2))
    # idx, proj = sli.interp_decomp(aslinearoperator(A), k)
    # B = A[:,idx[:k]]
    # P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
    # Aapprox = np.dot(B, P)
    # print(la.norm(A - Aapprox, 2))


if __name__ == '__main__':
    main()
