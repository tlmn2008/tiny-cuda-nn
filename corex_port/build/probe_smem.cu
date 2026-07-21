#include <cstdio>
#include <cstdint>

__global__ void probe(unsigned long long* out) {
	__shared__ float s[64];
	s[threadIdx.x % 64] = threadIdx.x;
	__syncthreads();
	if (threadIdx.x == 0) {
		out[0] = reinterpret_cast<uintptr_t>(&s[0]);
	}
}

int main() {
	unsigned long long* d = nullptr;
	cudaMalloc(&d, sizeof(unsigned long long));
	probe<<<1, 64>>>(d);
	cudaError_t e = cudaDeviceSynchronize();
	unsigned long long p = 0;
	cudaMemcpy(&p, d, sizeof(p), cudaMemcpyDeviceToHost);
	printf("sync=%s  shared&s[0]=0x%llx  hi32=0x%08x lo32=0x%08x  (hi!=0 => 32-bit token truncation WRONG)\n",
		cudaGetErrorString(e), p, (unsigned)(p>>32), (unsigned)(p & 0xffffffffull));
	return 0;
}
