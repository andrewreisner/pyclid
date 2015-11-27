import pyopencl_blas as blas
import numpy as np
import pyopencl.array as cl_array

import pyclid.util as util
from pyclid.qr import iddr_qrpiv


def iddr_id(queue, m, n, a, krank, lst, rnorms):
    iddr_qrpiv(queue, m, n, a, krank, lst, rnorms)



def iddr_rid(queue, m, n, matvect, krank):
    id_srand = util.setup_rand(queue)

    blas.setup()

    dtype = 'float64'
    clx = cl_array.zeros(queue, m, dtype)
    rnorms = cl_array.Array(queue, n, dtype)
    lst = cl_array.Array(queue, n, dtype)
    proj = cl_array.Array(queue, (krank+2,n), dtype)


    l = krank + 2

    for i in range(l):
        id_srand(m, clx)
        matvect(clx, proj[i,:])


    iddr_id(queue, l, n, proj, krank, lst, rnorms)
    #idx, proj = pyclid_iddr_rid(m, n, matvect, k)

    blas.teardown()

    return lst, proj
