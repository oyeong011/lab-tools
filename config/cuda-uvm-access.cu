#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

enum Pattern {
  PATTERN_LS = 0,
  PATTERN_HCHI = 1,
  PATTERN_HCLI = 2,
  PATTERN_LC = 3,
};

static int parse_pattern(const char *name) {
  if (std::strcmp(name, "ls") == 0) return PATTERN_LS;
  if (std::strcmp(name, "hchi") == 0) return PATTERN_HCHI;
  if (std::strcmp(name, "hcli") == 0) return PATTERN_HCLI;
  if (std::strcmp(name, "lc") == 0) return PATTERN_LC;
  return -1;
}

static const char *pattern_name(int pattern) {
  switch (pattern) {
    case PATTERN_LS: return "ls";
    case PATTERN_HCHI: return "hchi";
    case PATTERN_HCLI: return "hcli";
    case PATTERN_LC: return "lc";
    default: return "unknown";
  }
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

__global__ void uvm_access_kernel(float *data, size_t n, size_t ops,
                                  int pattern, float *sink) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;
  const size_t page_stride = 4096 / sizeof(float);
  const size_t low_coverage_n = n > 16 ? n / 16 : n;
  float local = 0.0f;

  for (size_t logical = tid; logical < ops; logical += stride) {
    size_t idx;
    if (pattern == PATTERN_LS) {
      idx = logical % n;
    } else if (pattern == PATTERN_HCHI) {
      idx = mix64(logical) % n;
    } else if (pattern == PATTERN_HCLI) {
      idx = (logical * page_stride) % n;
    } else {
      idx = logical % low_coverage_n;
    }
    float v = data[idx];
    v = v * 1.000001f + 1.0f;
    data[idx] = v;
    local += v;
  }

  atomicAdd(sink, local);
}

int main(int argc, char **argv) {
  const size_t mb = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 512;
  const char *pattern_arg = argc > 2 ? argv[2] : "ls";
  const int passes = argc > 3 ? std::atoi(argv[3]) : 2;
  const int pattern = parse_pattern(pattern_arg);
  if (pattern < 0 || passes <= 0 || mb == 0) {
    std::fprintf(stderr,
                 "Usage: cuda-uvm-access MANAGED_MB PATTERN PASSES\n"
                 "PATTERN must be one of: ls, hchi, hcli, lc\n");
    return 2;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  const size_t bytes = mb * 1024ULL * 1024ULL;
  const size_t n = bytes / sizeof(float);
  size_t ops_per_pass = n;
  if (pattern == PATTERN_HCLI) {
    ops_per_pass = n / (4096 / sizeof(float));
    if (ops_per_pass == 0) ops_per_pass = 1;
  } else if (pattern == PATTERN_LC) {
    ops_per_pass = n / 16;
    if (ops_per_pass == 0) ops_per_pass = n;
  }
  const size_t ops = ops_per_pass * static_cast<size_t>(passes);

  float *data = nullptr;
  float *sink = nullptr;
  CHECK_CUDA(cudaMallocManaged(&data, bytes));
  CHECK_CUDA(cudaMallocManaged(&sink, sizeof(float)));
  *sink = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<float>(i & 0xff) * 0.001f;
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  const int block = 256;
  int grid = prop.multiProcessorCount * 8;
  if (grid < 1) grid = 1;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  uvm_access_kernel<<<grid, block>>>(data, n, ops, pattern, sink);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  const double seconds = ms / 1000.0;
  const double touched_bytes = static_cast<double>(ops) * sizeof(float) * 2.0;
  const double gbps = touched_bytes / seconds / 1.0e9;

  double max_error = 0.0;
  const size_t sample_count = n < 1024 ? n : 1024;
  for (size_t i = 0; i < sample_count; ++i) {
    if (!std::isfinite(data[i])) {
      max_error = 1.0;
      break;
    }
  }

  std::printf("Device: %s\n", prop.name);
  std::printf("UVM pattern: %s\n", pattern_name(pattern));
  std::printf("Managed size: %zu MB\n", mb);
  std::printf("Elements: %zu\n", n);
  std::printf("Operations: %zu\n", ops);
  std::printf("Passes: %d\n", passes);
  std::printf("Kernel time: %.6f s\n", seconds);
  std::printf("Device approx bandwidth: %.6f GB/s\n", gbps);
  std::printf("Sink: %.6f\n", static_cast<double>(*sink));
  std::printf("Max error: %.6f\n", max_error);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(data));
  CHECK_CUDA(cudaFree(sink));
  return 0;
}
