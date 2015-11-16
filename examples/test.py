import pyclid
import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyopencl_blas as blas

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

rng = np.random.RandomState(1)  # change the seed to see different data
n = 5

dtype = 'float64'
A = np.zeros((n,n), dtype=dtype)

A[...] = rng.uniform(-1, 1, size=A.shape)


mv = pyclid.util.create_mvt(queue, A)
pyclid.iddr_rid(queue, n, n, mv, 4)
