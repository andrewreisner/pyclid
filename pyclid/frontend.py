from pyclid.types import LinearOperator
import pyclid.backend as backend


def interp_decomp(queue, A, k):
    if isinstance(A, LinearOperator):
        m, n = A.shape
        matvect = A.rmatvec
        idx, proj = backend.iddr_rid(queue, m, n, matvect, k)
        idx = idx.get()
        proj = proj.get()[:k, :(n-k)]
        return idx, proj
    else:
        raise TypeError('invalid input type (must be LinearOperator)')
