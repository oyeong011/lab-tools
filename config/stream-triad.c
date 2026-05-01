#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  size_t n = 33554432; /* 256 MiB per array of doubles */
  int repeats = 5;
  if (argc > 1) n = strtoull(argv[1], NULL, 10);
  if (argc > 2) repeats = atoi(argv[2]);
  if (n < 1024 || repeats < 1) {
    fprintf(stderr, "Usage: stream-triad [elements>=1024] [repeats>=1]\n");
    return 2;
  }

  double *a = aligned_alloc(64, n * sizeof(double));
  double *b = aligned_alloc(64, n * sizeof(double));
  double *c = aligned_alloc(64, n * sizeof(double));
  if (!a || !b || !c) {
    fprintf(stderr, "allocation failed for %zu elements\n", n);
    return 1;
  }

  #pragma omp parallel for
  for (size_t i = 0; i < n; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }

  double best = 1e99;
  double scalar = 3.0;
  for (int r = 0; r < repeats; r++) {
    double t0 = now_seconds();
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      c[i] = a[i] + scalar * b[i];
    }
    double elapsed = now_seconds() - t0;
    if (elapsed < best) best = elapsed;
  }

  double checksum = 0.0;
  for (size_t i = 0; i < n; i += n / 1024) {
    checksum += c[i];
  }
  double bytes = (double)n * sizeof(double) * 3.0;
  printf("Benchmark: STREAM triad\n");
  printf("Elements: %zu\n", n);
  printf("Repeats: %d\n", repeats);
  printf("OpenMP threads: %d\n", omp_get_max_threads());
  printf("Best time: %.6f s\n", best);
  printf("Triad bandwidth: %.2f MB/s\n", bytes / best / 1e6);
  printf("Sample checksum: %.6f\n", checksum);

  free(c);
  free(b);
  free(a);
  return 0;
}
