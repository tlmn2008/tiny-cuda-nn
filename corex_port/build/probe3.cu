#include <cstdio>
__global__ void k(unsigned* o) {
	o[threadIdx.x] = __nvvm_read_ptx_sreg_laneid();
	o[64 + threadIdx.x] = threadIdx.x & (warpSize - 1);
}
int main() {
	unsigned* d; cudaMalloc(&d, 128*sizeof(unsigned));
	k<<<1,64>>>(d);
	cudaError_t e = cudaDeviceSynchronize();
	unsigned h[128]; cudaMemcpy(h, d, 128*sizeof(unsigned), cudaMemcpyDeviceToHost);
	printf("sync=%s builtin[0,1,63]=%u,%u,%u mask[0,1,63]=%u,%u,%u\n",
		cudaGetErrorString(e), h[0],h[1],h[63], h[64],h[65],h[127]);
	return 0;
}
