#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void gemv_kernel(const float *a, const float *x, float *y, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  float sum = 0.0f;
  const float *base = a + static_cast<size_t>(row) * n;
  for (int col = 0; col < n; ++col) {
    sum += base[col] * x[col];
  }
  y[row] = sum;
}

__global__ void spmv_csr_kernel(const int *row_ptr, const int *col_idx,
                                const float *values, const float *x, float *y,
                                int rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    sum += values[p] * x[col_idx[p]];
  }
  y[row] = sum;
}

__global__ void gcn_agg_kernel(const int *row_ptr, const int *col_idx,
                               const float *features, float *out,
                               int rows, int feature_dim) {
  int row = blockIdx.x;
  int feat = threadIdx.x;
  if (row >= rows || feat >= feature_dim) return;
  float sum = 0.0f;
  for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    int src = col_idx[p];
    sum += features[static_cast<size_t>(src) * feature_dim + feat];
  }
  out[static_cast<size_t>(row) * feature_dim + feat] = sum;
}

static float gemv_value(int row, int col) {
  return static_cast<float>(((row * 17 + col * 13) & 0xff) + 1) * 0.0001f;
}

static float spmv_value(int row, int k) {
  return static_cast<float>(((row * 31 + k * 7) & 0x3f) + 1) * 0.001f;
}

static void usage() {
  std::fprintf(stderr,
               "Usage:\n"
               "  cuda-memory-kernels gemv N\n"
               "  cuda-memory-kernels spmv ROWS NNZ_PER_ROW\n"
               "  cuda-memory-kernels gcn ROWS DEGREE FEATURE_DIM\n");
}

static bool parse_positive_int(const char *text, int *out) {
  char *end = nullptr;
  errno = 0;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
      value > INT_MAX) {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

static bool product_fits_int(int a, int b) {
  return static_cast<unsigned long long>(a) *
             static_cast<unsigned long long>(b) <=
         static_cast<unsigned long long>(INT_MAX);
}

static int check_device_memory(size_t required_bytes) {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error cudaMemGetInfo: %s\n",
                 cudaGetErrorString(err));
    return 1;
  }
  if (required_bytes > (free_bytes * 9) / 10) {
    std::fprintf(stderr,
                 "Requested device allocation is too large: need %zu bytes, "
                 "free %zu bytes.\n",
                 required_bytes, free_bytes);
    return 2;
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const bool is_gemv = std::strcmp(argv[1], "gemv") == 0;
  const bool is_spmv = std::strcmp(argv[1], "spmv") == 0;
  const bool is_gcn = std::strcmp(argv[1], "gcn") == 0;
  if (!is_gemv && !is_spmv && !is_gcn) {
    usage();
    return 2;
  }

  int device = 0;
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const int block = 256;
  float max_error = 0.0f;
  double seconds = 0.0;
  double gflops = 0.0;
  double gbps = 0.0;

  if (is_gemv) {
    int n = 0;
    if (!parse_positive_int(argv[2], &n)) {
      usage();
      return 2;
    }
    const size_t matrix_elems = static_cast<size_t>(n) * n;
    const size_t device_bytes =
        (matrix_elems + static_cast<size_t>(n) + static_cast<size_t>(n)) *
        sizeof(float);
    int mem_status = check_device_memory(device_bytes);
    if (mem_status != 0) return mem_status;
    std::vector<float> h_a(matrix_elems);
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 0.0f);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        h_a[static_cast<size_t>(r) * n + c] = gemv_value(r, c);
      }
    }

    float *d_a = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, matrix_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), matrix_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), static_cast<size_t>(n) * sizeof(float), cudaMemcpyHostToDevice));

    const int grid = (n + block - 1) / block;
    CHECK_CUDA(cudaEventRecord(start));
    gemv_kernel<<<grid, block>>>(d_a, d_x, d_y, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    seconds = ms / 1000.0;
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));

    const int samples = n < 32 ? n : 32;
    for (int r = 0; r < samples; ++r) {
      float ref = 0.0f;
      for (int c = 0; c < n; ++c) ref += gemv_value(r, c);
      max_error = fmaxf(max_error, fabsf(ref - h_y[r]));
    }
    const double ops = 2.0 * static_cast<double>(n) * static_cast<double>(n);
    const double bytes = static_cast<double>(matrix_elems + n + n) * sizeof(float);
    gflops = ops / seconds / 1.0e9;
    gbps = bytes / seconds / 1.0e9;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    std::printf("Kernel: cuda-gemv\n");
    std::printf("Rows: %d\n", n);
    std::printf("Cols: %d\n", n);
  } else if (is_spmv) {
    int rows = 0;
    int nnz_per_row = 16;
    if (!parse_positive_int(argv[2], &rows) ||
        (argc > 3 && !parse_positive_int(argv[3], &nnz_per_row))) {
      usage();
      return 2;
    }
    if (!product_fits_int(rows, nnz_per_row)) {
      std::fprintf(stderr, "ROWS * NNZ_PER_ROW exceeds CSR int offset range.\n");
      return 2;
    }
    const int cols = rows;
    const size_t nnz = static_cast<size_t>(rows) * nnz_per_row;
    const size_t device_bytes =
        static_cast<size_t>(rows + 1) * sizeof(int) + nnz * sizeof(int) +
        nnz * sizeof(float) + static_cast<size_t>(cols) * sizeof(float) +
        static_cast<size_t>(rows) * sizeof(float);
    int mem_status = check_device_memory(device_bytes);
    if (mem_status != 0) return mem_status;
    std::vector<int> h_row_ptr(rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<float> h_values(nnz);
    std::vector<float> h_x(cols, 1.0f);
    std::vector<float> h_y(rows, 0.0f);
    for (int r = 0; r <= rows; ++r) h_row_ptr[r] = r * nnz_per_row;
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < nnz_per_row; ++k) {
        size_t p = static_cast<size_t>(r) * nnz_per_row + k;
        h_col_idx[p] = static_cast<int>(
            (static_cast<unsigned long long>(r) * 1315423911ULL +
             static_cast<unsigned long long>(k) * 2654435761ULL) %
            static_cast<unsigned long long>(cols));
        h_values[p] = spmv_value(r, k);
      }
    }

    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    float *d_values = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_row_ptr, static_cast<size_t>(rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(cols) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(rows) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), static_cast<size_t>(rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), static_cast<size_t>(cols) * sizeof(float), cudaMemcpyHostToDevice));

    const int grid = (rows + block - 1) / block;
    CHECK_CUDA(cudaEventRecord(start));
    spmv_csr_kernel<<<grid, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    seconds = ms / 1000.0;
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost));

    const int samples = rows < 32 ? rows : 32;
    for (int r = 0; r < samples; ++r) {
      float ref = 0.0f;
      for (int k = 0; k < nnz_per_row; ++k) ref += spmv_value(r, k);
      max_error = fmaxf(max_error, fabsf(ref - h_y[r]));
    }
    const double ops = 2.0 * static_cast<double>(nnz);
    const double bytes = static_cast<double>(nnz) * (sizeof(float) + sizeof(int) + sizeof(float)) +
                         static_cast<double>(rows + 1) * sizeof(int) +
                         static_cast<double>(rows) * sizeof(float);
    gflops = ops / seconds / 1.0e9;
    gbps = bytes / seconds / 1.0e9;

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    std::printf("Kernel: cuda-spmv\n");
    std::printf("Rows: %d\n", rows);
    std::printf("Cols: %d\n", cols);
    std::printf("NNZ per row: %d\n", nnz_per_row);
    std::printf("NNZ: %zu\n", nnz);
  } else {
    int rows = 0;
    int degree = 16;
    int feature_dim = 16;
    if (!parse_positive_int(argv[2], &rows) ||
        (argc > 3 && !parse_positive_int(argv[3], &degree)) ||
        (argc > 4 && !parse_positive_int(argv[4], &feature_dim))) {
      usage();
      return 2;
    }
    if (feature_dim > prop.maxThreadsPerBlock) {
      std::fprintf(stderr,
                   "FEATURE_DIM exceeds maxThreadsPerBlock on this device.\n");
      return 2;
    }
    if (rows > prop.maxGridSize[0]) {
      std::fprintf(stderr, "ROWS exceeds max grid x dimension on this device.\n");
      return 2;
    }
    if (!product_fits_int(rows, degree)) {
      std::fprintf(stderr, "ROWS * DEGREE exceeds CSR int offset range.\n");
      return 2;
    }
    const size_t edges = static_cast<size_t>(rows) * degree;
    const size_t feature_elems = static_cast<size_t>(rows) * feature_dim;
    const size_t device_bytes =
        static_cast<size_t>(rows + 1) * sizeof(int) + edges * sizeof(int) +
        feature_elems * sizeof(float) + feature_elems * sizeof(float);
    int mem_status = check_device_memory(device_bytes);
    if (mem_status != 0) return mem_status;
    std::vector<int> h_row_ptr(rows + 1);
    std::vector<int> h_col_idx(edges);
    std::vector<float> h_features(feature_elems);
    std::vector<float> h_out(feature_elems, 0.0f);
    for (int r = 0; r <= rows; ++r) h_row_ptr[r] = r * degree;
    for (int r = 0; r < rows; ++r) {
      for (int k = 0; k < degree; ++k) {
        size_t p = static_cast<size_t>(r) * degree + k;
        h_col_idx[p] = static_cast<int>(
            (static_cast<unsigned long long>(r) * 1103515245ULL +
             static_cast<unsigned long long>(k) * 12345ULL) %
            static_cast<unsigned long long>(rows));
      }
      for (int f = 0; f < feature_dim; ++f) {
        h_features[static_cast<size_t>(r) * feature_dim + f] =
            static_cast<float>(((r * 19 + f * 23) & 0xff) + 1) * 0.001f;
      }
    }

    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    float *d_features = nullptr;
    float *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_row_ptr, static_cast<size_t>(rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, edges * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_features, feature_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, feature_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), static_cast<size_t>(rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, h_col_idx.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_features, h_features.data(), feature_elems * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid(rows);
    const dim3 threads(feature_dim);
    CHECK_CUDA(cudaEventRecord(start));
    gcn_agg_kernel<<<grid, threads>>>(d_row_ptr, d_col_idx, d_features, d_out, rows, feature_dim);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    seconds = ms / 1000.0;
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, feature_elems * sizeof(float), cudaMemcpyDeviceToHost));

    const int sample_rows = rows < 64 ? rows : 64;
    const int sample_feats = feature_dim < 32 ? feature_dim : 32;
    for (int sr = 0; sr < sample_rows; ++sr) {
      int r = static_cast<int>((static_cast<long long>(sr) * 9973LL) % rows);
      for (int sf = 0; sf < sample_feats; ++sf) {
        int f = static_cast<int>((static_cast<long long>(sf) * 37LL) %
                                 feature_dim);
        float ref = 0.0f;
        for (int k = 0; k < degree; ++k) {
          int src = h_col_idx[static_cast<size_t>(r) * degree + k];
          ref += h_features[static_cast<size_t>(src) * feature_dim + f];
        }
        float got = h_out[static_cast<size_t>(r) * feature_dim + f];
        max_error = fmaxf(max_error, fabsf(ref - got));
      }
    }
    const double ops = static_cast<double>(edges) * static_cast<double>(feature_dim);
    const double bytes = static_cast<double>(edges) * sizeof(int) +
                         static_cast<double>(edges) * static_cast<double>(feature_dim) * sizeof(float) +
                         static_cast<double>(feature_elems) * sizeof(float);
    gflops = ops / seconds / 1.0e9;
    gbps = bytes / seconds / 1.0e9;

    CHECK_CUDA(cudaFree(d_row_ptr));
    CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_features));
    CHECK_CUDA(cudaFree(d_out));
    std::printf("Kernel: cuda-gcn-agg\n");
    std::printf("Rows: %d\n", rows);
    std::printf("Degree: %d\n", degree);
    std::printf("Feature dim: %d\n", feature_dim);
    std::printf("Edges: %zu\n", edges);
  }

  std::printf("Device: %s\n", prop.name);
  std::printf("Kernel time: %.6f s\n", seconds);
  std::printf("Device approx bandwidth: %.6f GB/s\n", gbps);
  std::printf("Device approx GFLOP/s: %.6f\n", gflops);
  std::printf("Max error: %.6f\n", static_cast<double>(max_error));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}
