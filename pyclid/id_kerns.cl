__kernel void rnorm(__global double *a, __global double *rnorms, int stride)
{
	int i = get_global_id(0);

	rnorms[i] = a[i + i*stride];
}


__kernel void moveup(__global double *a, int k, int stride)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	a[i + j*stride] = a[i + (k+j)*stride];
}
