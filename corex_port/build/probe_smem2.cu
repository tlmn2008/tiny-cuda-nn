#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uintptr_t smem_base() {
	__shared__ int anchor;
	return reinterpret_cast<uintptr_t>(&anchor) & 0xffffffff00000000ull;
}

__global__ void probe(unsigned long long* out) {
	__shared__ float a[64];
	__shared__ double b[32];
	uintptr_t pa = reinterpret_cast<uintptr_t>(&a[7]);
	uintptr_t pb = reinterpret_cast<uintptr_t>(&b[3]);
	unsigned offa = (unsigned)(pa & 0xffffffffu);
	uintptr_t base = smem_base();
	// reconstruct a[7] from base|offset, write a sentinel through it, read back via real ptr
	float* recon = reinterpret_cast<float*>(base | (uintptr_t)offa);
	*recon = 3.14159f;
	out[0] = pa;
	out[1] = pb;
	out[2] = base;
	out[3] = (unsigned long long)(base | (uintptr_t)offa); // should equal pa
	out[4] = reinterpret_cast<unsigned long long&>(a[7]);  // sentinel bits (float 3.14159)
	out[5] = (pa & 0xffffffff00000000ull) == (pb & 0xffffffff00000000ull); // same base?
}

int main() {
	unsigned long long* d = nullptr;
	cudaMalloc(&d, 8 * sizeof(unsigned long long));
	probe<<<1, 1>>>(d);
	cudaError_t e = cudaDeviceSynchronize();
	unsigned long long h[8] = {0};
	cudaMemcpy(h, d, 8 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	float sentinel; unsigned s32 = (unsigned)h[4]; sentinel = reinterpret_cast<float&>(s32);
	printf("sync=%s\n", cudaGetErrorString(e));
	printf("a7=0x%llx b3=0x%llx base=0x%llx recon=0x%llx  recon==a7? %s\n",
		h[0], h[1], h[2], h[3], (h[3]==h[0])?"YES":"NO");
	printf("sentinel written via recon, read via real a[7] = %f (expect 3.14159)\n", sentinel);
	printf("a and b share base? %s\n", h[5]?"YES":"NO");
	return 0;
}
