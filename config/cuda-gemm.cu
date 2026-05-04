// CUDA SGEMM benchmark using cuBLAS — mirror of opencl-gemm.c.
// Reports kernel time and approximate GFLOP/s, matching the OpenCL kernel's
// stdout schema so summarize-suite/suite-compare can join the metrics.
//
// Build with: nvcc -O2 cuda-gemm.cu -lcublas -o cuda-gemm
// Run:        ./cuda-gemm [N]   (default N = 256, square SGEMM)
#include <cublas_v2.h>
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

#define CUBLAS_OK(call)                                                        \
  do {                                                                         \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                   static_cast<int>(st));                                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : 256;
  size_t bytes = sizeof(float) * static_cast<size_t>(n) * static_cast<size_t>(n);
  std::vector<float> hA(static_cast<size_t>(n) * n), hB(static_cast<size_t>(n) * n), hC(static_cast<size_t>(n) * n);
  for (size_t i = 0; i < hA.size(); ++i) {
    hA[i] = static_cast<float>((i * 13) % 17) * 0.01f;
    hB[i] = static_cast<float>((i * 7) % 19) * 0.01f;
  }

  float *dA, *dB, *dC;
  CUDA_OK(cudaMalloc(&dA, bytes));
  CUDA_OK(cudaMalloc(&dB, bytes));
  CUDA_OK(cudaMalloc(&dC, bytes));
  CUDA_OK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_OK(cublasCreate(&handle));
  float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  CUBLAS_OK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n));
  CUDA_OK(cudaDeviceSynchronize());

  cudaEventRecord(e0);
  CUBLAS_OK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n, dB, n, &beta, dC, n));
  cudaEventRecord(e1);
  CUDA_OK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, e0, e1);

  CUDA_OK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

  double kernel_s = ms / 1000.0;
  double flops = 2.0 * static_cast<double>(n) * n * n;
  double gflops = flops / kernel_s / 1e9;

  float sample_max_err = 0.0f;
  for (int i = 0; i < std::min(n, 8); ++i) {
    float ref = 0.0f;
    for (int k = 0; k < n; ++k) ref += hA[i + k * n] * hB[k + 0 * n];
    sample_max_err = std::max(sample_max_err, std::fabs(hC[i + 0 * n] - ref));
  }

  std::printf("N: %d\n", n);
  std::printf("Kernel time: %.6f s\n", kernel_s);
  std::printf("Device approx GFLOP/s: %.2f\n", gflops);
  std::printf("Sample max error: %.6g\n", sample_max_err);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return 0;
}
