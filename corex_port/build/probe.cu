#include <cstdio>
#include <cuda_fp16.h>

__global__ void k(int* out) {
#ifdef __CUDA_ARCH__
	out[0] = __CUDA_ARCH__;
#else
	out[0] = -1;
#endif
#ifdef __ILUVATAR__
	out[1] = 1;
#else
	out[1] = 0;
#endif
	__half2 a = __half2(__float2half(1.0f), __float2half(2.0f));
	__half2 b = __hadd2(a, a);
	out[2] = (int)__low2float(b);
	out[3] = warpSize;
}

int main() {
	int* d; cudaMalloc(&d, 8*sizeof(int));
	k<<<1,1>>>(d);
	cudaError_t e = cudaDeviceSynchronize();
	int h[8]; cudaMemcpy(h, d, 8*sizeof(int), cudaMemcpyDeviceToHost);
	printf("sync=%s __CUDA_ARCH__=%d __ILUVATAR__=%d hadd2_low=%d warpSize=%d\n",
		cudaGetErrorString(e), h[0], h[1], h[2], h[3]);
	return 0;
}
