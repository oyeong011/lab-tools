#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(expr) do { \
  cl_int _err = (expr); \
  if (_err != CL_SUCCESS) { \
    fprintf(stderr, "%s failed: %d\n", #expr, _err); \
    return 1; \
  } \
} while (0)

static const char *kernel_source =
"__kernel void gemm(const int n, __global const float *a, __global const float *b, __global float *c) {\n"
"  int row = get_global_id(0);\n"
"  int col = get_global_id(1);\n"
"  if (row >= n || col >= n) return;\n"
"  float sum = 0.0f;\n"
"  for (int k = 0; k < n; k++) {\n"
"    sum += a[row * n + k] * b[k * n + col];\n"
"  }\n"
"  c[row * n + col] = sum;\n"
"}\n";

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int verify_sample(const float *a, const float *b, const float *c, int n) {
  double max_error = 0.0;
  int step = n / 8;
  if (step < 1) step = 1;
  for (int row = 0; row < n; row += step) {
    for (int col = 0; col < n; col += step) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += (double)a[row * n + k] * (double)b[k * n + col];
      }
      double error = fabs((double)c[row * n + col] - sum);
      if (error > max_error) max_error = error;
    }
  }
  printf("Sample max error: %.6g\n", max_error);
  return max_error < 0.01 ? 0 : 1;
}

int main(int argc, char **argv) {
  int n = 256;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
  if (n < 16 || n > 2048) {
    fprintf(stderr, "Matrix size must be between 16 and 2048.\n");
    return 2;
  }

  const size_t elements = (size_t)n * (size_t)n;
  const size_t bytes = elements * sizeof(float);
  cl_int err = CL_SUCCESS;
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;

  CHECK(clGetPlatformIDs(1, &platform, NULL));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

  char name[256] = {0};
  CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL));
  printf("Device: %s\n", name);
  printf("Matrix size: %d x %d\n", n, n);

  float *a = malloc(bytes);
  float *b = malloc(bytes);
  float *c = calloc(elements, sizeof(float));
  if (!a || !b || !c) {
    fprintf(stderr, "host allocation failed\n");
    return 1;
  }

  for (size_t i = 0; i < elements; i++) {
    a[i] = (float)(((int)(i % 97)) - 48) / 97.0f;
    b[i] = (float)(((int)(i % 89)) - 44) / 89.0f;
  }

  cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK(err);
  cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK(err);
  cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_source, NULL, &err);
  CHECK(err);

  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[8192] = {0};
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    fprintf(stderr, "build failed: %d\n%s\n", err, log);
    return 1;
  }

  cl_kernel kernel = clCreateKernel(program, "gemm", &err);
  CHECK(err);
  cl_mem da = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a, &err);
  CHECK(err);
  cl_mem db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b, &err);
  CHECK(err);
  cl_mem dc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
  CHECK(err);

  CHECK(clSetKernelArg(kernel, 0, sizeof(n), &n));
  CHECK(clSetKernelArg(kernel, 1, sizeof(da), &da));
  CHECK(clSetKernelArg(kernel, 2, sizeof(db), &db));
  CHECK(clSetKernelArg(kernel, 3, sizeof(dc), &dc));

  const size_t rounded = ((size_t)n + 15) / 16 * 16;
  const size_t global[2] = {rounded, rounded};
  const size_t local[2] = {16, 16};
  cl_event kernel_event = NULL;
  double t0 = now_seconds();
  CHECK(clEnqueueNDRangeKernel(q, kernel, 2, NULL, global, local, 0, NULL, &kernel_event));
  CHECK(clFinish(q));
  double t1 = now_seconds();
  cl_ulong event_start = 0;
  cl_ulong event_end = 0;
  CHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(event_start), &event_start, NULL));
  CHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(event_end), &event_end, NULL));

  CHECK(clEnqueueReadBuffer(q, dc, CL_TRUE, 0, bytes, c, 0, NULL, NULL));

  double elapsed = t1 - t0;
  double device_elapsed = (double)(event_end - event_start) / 1e9;
  double gflops = (2.0 * (double)n * (double)n * (double)n) / elapsed / 1e9;
  double device_gflops = (2.0 * (double)n * (double)n * (double)n) / device_elapsed / 1e9;
  printf("Kernel time: %.6f s\n", elapsed);
  printf("Device kernel time: %.6f s\n", device_elapsed);
  printf("Approx GFLOP/s: %.2f\n", gflops);
  printf("Device approx GFLOP/s: %.2f\n", device_gflops);
  int ok = verify_sample(a, b, c, n);

  clReleaseMemObject(dc);
  clReleaseMemObject(db);
  clReleaseMemObject(da);
  clReleaseKernel(kernel);
  clReleaseEvent(kernel_event);
  clReleaseProgram(program);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);
  free(c);
  free(b);
  free(a);
  return ok;
}
