import pyopencl_blas as blas
import numpy as np
import pyopencl.array as cl_array
from pyopencl import clrandom
import time


def id_srand(queue, m, x):
    gen = clrandom.RanluxGenerator(queue, seed=time.time())
    gen.fill_uniform(x)
#def yvec(queue, m, n, matvectk, 

def iddr_rid(queue, m, n, matvect, k):
    blas.setup()

    dtype = 'float64'
    x = np.ones(n, dtype=dtype)
    clx = cl_array.zeros(queue, x.shape, x.dtype)
    cly = cl_array.Array(queue, m, x.dtype)
    #clx.set(x)

    for i in range(10):
        id_srand(queue, m, clx)
        #time.sleep(1)
        print clx.get()

    matvect(clx, cly)
    #print cly.get()

    #idx, proj = pyclid_iddr_rid(m, n, matvect, k)

    blas.teardown()
