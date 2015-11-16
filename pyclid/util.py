import pyopencl_blas as blas
from pyopencl.array import Array


def create_mvt(queue, A):
    cla = Array(queue, A.shape, A.dtype)

    cla.set(A)

    def matvect(x, y):
        blas.gemv(queue, cla, x, y, transA=True)
        return

    return matvect
