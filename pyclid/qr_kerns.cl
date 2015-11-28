inline void reduce_group_sum(__local double *ss_l)
{
	int lid = get_local_id(0);
	int group_size = get_local_size(0);

	// Naively round down to nearest power of 2
	int pwt = 1;
	while (pwt <= group_size) pwt <<= 1;
	pwt >>= 1;

	barrier(CLK_LOCAL_MEM_FENCE);
	// preprocess: sum values beyond pwt into beginning of array
	if ((lid+1) > pwt) {
		ss_l[(lid+1)^pwt] += ss_l[lid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = pwt/2; i > 0; i >>= 1) {
		if (lid < i) {
			ss_l[lid] += ss_l[lid + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel void ss(__global double *a_g, __global double *ss_g, __local double *ss_l,
                 int k, int stride) {
	int lid = get_local_id(0);
	int gid = get_group_id(1);

	int idx = (k + gid) + lid*stride;
	int ss_loc = a_g[idx]*a_g[idx];
	ss_l[lid] = a_g[idx]*a_g[idx];

	reduce_group_sum(ss_l);

	if (lid == 0) {
		ss_g[gid+k] = ss_l[lid];
	}
}


__kernel void swap_col(__global double *a, __global double *ss, __global double *r,
                       int c1, int c2, int stride)
{
	int lid = get_local_id(0);
	double tmp = a[lid*stride + c1];

	a[lid*stride + c1] = a[lid*stride + c2];
	a[lid*stride + c2] = tmp;

	// wasteful
	tmp = r[lid*stride + c1];
	r[lid*stride + c1] = r[lid*stride + c2];
	r[lid*stride + c2] = tmp;

	tmp = ss[c1];
	ss[c1] = ss[c2];
	ss[c2] = tmp;
}


__kernel void proj_rm(__global double *a_g, __global double *r_g,
                      __global double *ss,
                      __local double *qk, __local double *a_l,
                      __local double *aj_qk, int k, int stride)
{
	int lid = get_local_id(0);
	int gid = get_group_id(1);
	int ki = k + lid*stride;
	int jdx = (k + gid + 1) + lid*stride;


	double rkk = sqrt(ss[k]);
	if (get_global_id(0) == 0 && get_global_id(1) == 0) {
		r_g[k + k*stride] = rkk;
	}
	qk[lid] = a_g[ki] / rkk;
	a_l[lid] = a_g[jdx];
	aj_qk[lid] = qk[lid]*a_l[lid];

	reduce_group_sum(aj_qk);

	double r_kj = aj_qk[0];
	if (lid == 0) {
		r_g[k*stride + (k + gid + 1)] = r_kj;
	}

	a_g[jdx] = a_l[lid] - r_kj*qk[lid];
}


__kernel void norm(__global double *r_g, __global double *ss,
                   int k, int stride)
{
	/* Doesn't do much now, anticipating the potential need
	   for Q */
	if (get_global_id(0) == 0 && get_global_id(1) == 0) {
		r_g[k+k*stride] = sqrt(ss[k]);
	}
}
