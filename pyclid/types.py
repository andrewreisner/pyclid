import numpy as np

class LinearOperator:

    def __init__(self, shape, matvec, rmatvec=None, dtype=np.float64):
        self.shape = tuple(shape)
        self.matvec = matvec

        if rmatvec is not None:
            self.rmatvec = rmatvec
        else:
            def rmatvec(v):
                raise NotImplementedError('rmatvec is not defined')
            self.rmatvec = rmatvec

        self.dtype = dtype
