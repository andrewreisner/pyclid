#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la

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

    return A #np.asfortranarray(A)


def main():
    ctx = cl.create_some_context()
    # devices = ctx.get_info(cl.context_info.DEVICES)
    # print(devices[0].get_info(cl.device_info.VERSION))
    queue = cl.CommandQueue(ctx)

    dtype = 'float64'
    n = 100
    krank = 20

    A = setup_lowrank(n, dtype=dtype)
    mvt = pyclid.util.setup_matvect(queue, A)

    print('finished setup')
    pyclid.iddr_rid(queue, n, n, mvt, krank)


if __name__ == '__main__':
    main()
