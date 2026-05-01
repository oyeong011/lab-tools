#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  int n = 1024;
  int repeats = 3;
  if (argc > 1) n = atoi(argv[1]);
  if (argc > 2) repeats = atoi(argv[2]);
  if (n < 16 || repeats < 1) {
    fprintf(stderr, "Usage: openblas-gemm [n>=16] [repeats>=1]\n");
    return 2;
  }

  size_t elements = (size_t)n * (size_t)n;
  double *a = aligned_alloc(64, elements * sizeof(double));
  double *b = aligned_alloc(64, elements * sizeof(double));
  double *c = aligned_alloc(64, elements * sizeof(double));
  if (!a || !b || !c) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }
  for (size_t i = 0; i < elements; i++) {
    a[i] = (double)(((int)(i % 97)) - 48) / 97.0;
    b[i] = (double)(((int)(i % 89)) - 44) / 89.0;
    c[i] = 0.0;
  }

  double best = 1e99;
  for (int r = 0; r < repeats; r++) {
    double t0 = now_seconds();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    double elapsed = now_seconds() - t0;
    if (elapsed < best) best = elapsed;
  }

  double checksum = 0.0;
  int step = n / 8;
  if (step < 1) step = 1;
  for (int row = 0; row < n; row += step) {
    for (int col = 0; col < n; col += step) {
      checksum += c[(size_t)row * n + col];
    }
  }

  double gflops = (2.0 * (double)n * (double)n * (double)n) / best / 1e9;
  printf("Benchmark: OpenBLAS DGEMM\n");
  printf("Matrix size: %d x %d\n", n, n);
  printf("Repeats: %d\n", repeats);
  printf("Best time: %.6f s\n", best);
  printf("OpenBLAS GFLOP/s: %.2f\n", gflops);
  printf("Sample checksum: %.6f\n", checksum);

  free(c);
  free(b);
  free(a);
  return 0;
}
