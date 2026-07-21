#include <cstdio>

__device__ unsigned laneid_esc() {
	unsigned r;
	asm volatile("mov.u32 %0, %%laneid;" : "=r"(r));
	return r;
}
__device__ unsigned laneid_builtin() {
#if defined(__has_builtin)
#endif
	return __nvvm_read_ptx_sreg_laneid();
}

__global__ void k(unsigned* o) {
	o[threadIdx.x] = laneid_esc();
	o[64 + threadIdx.x] = laneid_builtin();
}

int main() {
	unsigned* d; cudaMalloc(&d, 128*sizeof(unsigned));
	k<<<1,64>>>(d);
	cudaError_t e = cudaDeviceSynchronize();
	unsigned h[128]; cudaMemcpy(h, d, 128*sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("sync=%s esc[0,1,63]=%u,%u,%u builtin[0,1,63]=%u,%u,%u\n",
		cudaGetErrorString(e), h[0], h[1], h[63], h[64], h[65], h[127]);
	return 0;
}
