import time
import os

import numpy as np

import pyopencl_blas as blas
from pyopencl.array import Array
from pyopencl import clrandom
from pyopencl.reduction import ReductionKernel
import pyopencl as cl


def setup_rand(queue):
    gen = clrandom.RanluxGenerator(queue, seed=time.time())
    return lambda m, x: gen.fill_uniform(x)


def setup_matvect(queue, A):
    cla = Array(queue, A.shape, A.dtype)

    cla.set(A)

    def matvect(x, y):
        blas.gemv(queue, cla, x, y, transA=True)
        return

    return matvect


def get_source(fname):
    fpath = os.path.dirname(__file__)
    fh = open(fpath + '/' + fname, 'r')
    src = ''.join(fh.readlines())
    return src


def argmax(queue, a):
    dev = queue.get_info(cl.command_queue_info.DEVICE)
    ctx = queue.get_info(cl.command_queue_info.CONTEXT)
    dtype = np.dtype([
        ('cur_max', np.float64),
        ('cur_ind', np.int32)])
    dtype, c_decl = cl.tools.match_dtype_to_c_struct(dev, 'max_ind', dtype)
    dtype = cl.tools.get_or_register_dtype('max_ind', dtype)

    preamble = c_decl + '''
    max_ind mi_neutral()
    {
        max_ind res;
        res.cur_max = -(1<<30);
        res.cur_ind = 0;
        return res;
    }


    max_ind mi_from_scalar(double v, int i)
    {
        max_ind res;
        res.cur_max = v;
        res.cur_ind = i;
        return res;
    }


    max_ind mi_comp(max_ind a, max_ind b)
    {
        max_ind res = a;
        if (b.cur_max > res.cur_max) {
            res.cur_max = b.cur_max;
            res.cur_ind = b.cur_ind;
        }

        return res;
    }
    '''

    red = ReductionKernel(ctx, dtype,
                          neutral='mi_neutral()',
                          reduce_expr='mi_comp(a, b)',
                          map_expr='mi_from_scalar(x[i],i)',
                          arguments='__global double *x',
                          preamble=preamble)

    mi = red(a).get()
    return np.int32(mi['cur_ind'])
