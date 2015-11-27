inline void reduce_group_sum(__local double *ss_l)
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0);

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_size/2; i > 0; i >>= 1) {
		if (lid < i) {
			ss_l[lid] += ss_l[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel void ss(__global double *a_g, __global double *ss_g, __local double *ss_l) {
	int lid = get_local_id(0);
	int gid = get_group_id(1);

	int m = get_global_size(0);
	int n = get_global_size(1);

	int idx = gid + lid*n;
	int ss_loc = a_g[idx]*a_g[idx];
	ss_l[lid] = a_g[idx]*a_g[idx];

	reduce_group_sum(ss_l);

	if (lid == 0) {
		ss_g[gid] = ss_l[lid];
	}
}


__kernel void swap_col(__global double *a, int c1, int c2, int stride)
{
	int lid = get_local_id(0);
	double tmp = a[lid*stride + c1];

	a[lid*stride + c1] = a[lid*stride + c2];
	a[lid*stride + c2] = tmp;
}


__kernel void proj_rm(__global double *a_g, __local double *qk, int k, int stride)
{
	int lid = get_local_id(0);
	int ki = k + lid*stride;

}

