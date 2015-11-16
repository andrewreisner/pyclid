import pyopencl_blas as blas
import numpy as np
from pyopencl.array import Array

def iddr_rid(queue, m, n, matvect, k):
    blas.setup()

    dtype = 'float64'
    x = np.ones(n, dtype=dtype)
    clx = Array(queue, x.shape, x.dtype)
    cly = Array(queue, m, x.dtype)
    clx.set(x)

    matvect(clx, cly)
    print cly.get()
    #idx, proj = pyclid_iddr_rid(m, n, matvect, k)

    blas.teardown()
