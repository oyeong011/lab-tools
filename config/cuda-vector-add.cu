// CUDA vector-add benchmark — mirror of opencl-vector-add.c so suite-compare can
// align iGPU (HD 630 / OpenCL) vs dGPU (RTX 5060 / CUDA) on the same kernel shape.
//
// Build with: nvcc -O2 cuda-vector-add.cu -o cuda-vector-add
// Run:        ./cuda-vector-add [N]   (default N = 1<<24)
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
  long n = (argc > 1) ? std::atol(argv[1]) : (1L << 24);
  std::vector<float> ha(n), hb(n), hc(n);
  for (long i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i) * 1e-3f;
    hb[i] = static_cast<float>(i) * 2e-3f;
  }

  float *da, *db, *dc;
  size_t bytes = sizeof(float) * static_cast<size_t>(n);
  CUDA_OK(cudaMalloc(&da, bytes));
  CUDA_OK(cudaMalloc(&db, bytes));
  CUDA_OK(cudaMalloc(&dc, bytes));

  auto t_h2d_0 = std::chrono::steady_clock::now();
  CUDA_OK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
  auto t_h2d_1 = std::chrono::steady_clock::now();

  cudaEvent_t k0, k1;
  cudaEventCreate(&k0);
  cudaEventCreate(&k1);
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  cudaEventRecord(k0);
  vector_add<<<grid, block>>>(da, db, dc, static_cast<int>(n));
  cudaEventRecord(k1);
  CUDA_OK(cudaEventSynchronize(k1));
  float kernel_ms = 0.0f;
  cudaEventElapsedTime(&kernel_ms, k0, k1);

  auto t_d2h_0 = std::chrono::steady_clock::now();
  CUDA_OK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
  auto t_d2h_1 = std::chrono::steady_clock::now();

  float max_err = 0.0f;
  for (long i = 0; i < n; ++i) {
    float ref = ha[i] + hb[i];
    max_err = std::max(max_err, std::fabs(hc[i] - ref));
  }

  double h2d_s = std::chrono::duration<double>(t_h2d_1 - t_h2d_0).count();
  double d2h_s = std::chrono::duration<double>(t_d2h_1 - t_d2h_0).count();
  double kernel_s = kernel_ms / 1000.0;
  double bytes_total = static_cast<double>(bytes) * 3.0;
  double dev_bw_gbs = bytes_total / kernel_s / 1e9;

  std::printf("N: %ld\n", n);
  std::printf("Kernel time: %.6f s\n", kernel_s);
  std::printf("H2D time: %.6f s\n", h2d_s);
  std::printf("D2H time: %.6f s\n", d2h_s);
  std::printf("Device approx bandwidth: %.2f GB/s\n", dev_bw_gbs);
  std::printf("Max error: %.6g\n", max_err);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaEventDestroy(k0);
  cudaEventDestroy(k1);
  return 0;
}
