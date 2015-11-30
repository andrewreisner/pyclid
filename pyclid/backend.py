import pyopencl_blas as blas
import numpy as np
import pyopencl.array as cl_array
import pyopencl as cl

import pyclid.util as util
from pyclid.qr import iddr_qrpiv


def idd_lssolve(queue, m, n, a, krank):
    for j in range(n - krank):
        blas.trsv(queue, a[:krank,:krank], a[:krank,krank+j],lower=False)

    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    prg = cl.Program(ctx, util.get_source('id_kerns.cl')).build()
    prg.moveup(queue, [krank, n-krank], None, a.data, np.int32(krank), np.int32(n))


def iddr_id(queue, m, n, a, krank, lst, rnorms):
    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    iddr_qrpiv(queue, m, n, a, krank, lst, rnorms)
    id_prg = cl.Program(ctx, util.get_source('id_kerns.cl')).build()
    id_prg.rnorm(queue, [krank, 1], None, a.data, rnorms.data, np.int32(n))

    idd_lssolve(queue, m, n, a, krank)


def iddr_rid(queue, m, n, matvect, krank):
    id_srand = util.setup_rand(queue)

    blas.setup()

    dtype = 'float64'
    clx = cl_array.zeros(queue, m, dtype)
    rnorms = cl_array.Array(queue, n, dtype)
    lst = cl_array.Array(queue, n, np.int32)
    proj = cl_array.Array(queue, (krank+2,n), dtype)


    l = krank + 2

    for i in range(l):
        id_srand(m, clx)
        matvect(clx, proj[i,:])


    iddr_id(queue, l, n, proj, krank, lst, rnorms)

    blas.teardown()

    return lst, proj
