import backend


def interp_decomp(A, k):
    if isinstance(A, LinearOperator):
        m, n = A.shape
        matvect = A.rmatvec
        idx, proj = backend.iddr_rid(m, n, matvect, k)
        return idx, proj
    else:
        raise TypeError('invalid input type (must be LinearOperator)')
