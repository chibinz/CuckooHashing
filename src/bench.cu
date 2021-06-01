#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"

#include "bench.h"
#include "device.h"
#include "multi.h"
#include "types.h"

using Table = MultilevelTable;

constexpr u32 repeat = 16;

inline double time_func(std::function<void()> f) {
  float duration;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  f();

  cudaEventCreate(&stop);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&duration, start, stop);
  return (double)(duration);
}

void test() {
  u32 capacity = 1024;
  u32 entry = 512;
  auto t = new Table(capacity);

  u32 *key, *set;
  cudaMalloc(&key, sizeof(u32) * entry);
  cudaMallocManaged(&set, sizeof(u32) * entry);
  cudaMemset(set, 0, sizeof(u32) * entry);
  randomizeDevice(key, entry);
  syncCheck();

  t->insert(key, entry);
  t->lookup(key, set, entry);

  syncCheck();

  for (u32 i = 0; i < entry; i += 1) {
    assert(set[i] == 1);
  }

  cudaFree(key);
  delete t;
}

void insertion() {
  printf("Insertion\n");
  printf("%-16s%-16s%-16s%-16s\n", "s", "Insertion/Mops", "Mean/ms",
         "StdDev/ms");

  for (u32 s = 10; s < 25; s += 1) {
    double sum = 0.0, sum2 = 0.0;
    u32 *key, n = 1 << s;
    cudaMalloc(&key, sizeof(u32) * n);
    randomizeDevice(key, n);
    syncCheck();

    for (u32 i = 0; i < repeat; i += 1) {
      auto t = new Table(1 << 25);

      auto dt = time_func([&] { t->insert(key, n); });
      sum += dt;
      sum2 += dt * dt;

      delete t;
      syncCheck();
    }

    cudaFree(key);

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16u%-16.4f%-16.4f%-16.4f\n", s, (n / 10e6) / (mean / 10e3), mean,
           stddev);
  }
}

void lookup() {
  printf("%s\n", "Lookup");
  printf("%-16s%-16s%-16s%-16s\n", "i", "Lookup/Mops", "Mean/ms", "StdDev/ms");

  u32 *key, *set, n = 1 << 24;
  cudaMalloc(&key, sizeof(u32) * n);
  cudaMalloc(&set, sizeof(u32) * n);
  randomizeDevice(key, n);
  syncCheck();

  auto t = new Table(1 << 25);
  t->insert(key, n);
  syncCheck();

  for (u32 i = 0; i <= 10; i += 1) {
    double sum = 0.0, sum2 = 0.0;
    if (i != 0) {
      randomizeDevice(key, n * i / 10);
    }
    cudaMemset(set, 0, sizeof(u32) * n);
    syncCheck();

    for (u32 j = 0; j < repeat; j += 1) {
      auto dt = time_func([&] { t->lookup(key, set, n); });
      sum += dt;
      sum2 += dt * dt;

      syncCheck();
    }

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16u%-16.4f%-16.4f%-16.4f\n", i, (n / 10e6) / (mean / 10e3), mean,
           stddev);
  }

  delete t;
  cudaFree(key);
}

void stress() {
  printf("%s\n", "Lookup");
  printf("%-16s%-16s%-16s%-16s\n", "i", "Lookup/Mops", "Mean/ms", "StdDev/ms");

  double scale[] = {2.0, 1.9, 1.8, 1.7,  1.6,  1.5, 1.4,
                    1.3, 1.2, 1.1, 1.05, 1.02, 1.01};

  u32 *key, *set, n = 1 << 24;
  cudaMalloc(&key, sizeof(u32) * n);
  cudaMalloc(&set, sizeof(u32) * n);
  randomizeDevice(key, n);
  syncCheck();

  auto t = new Table(1 << 25);
  t->insert(key, n);
  syncCheck();

  for (u32 i = 0; i <= 10; i += 1) {
    double sum = 0.0, sum2 = 0.0;
    if (i != 0) {
      randomizeDevice(key, n * i / 10);
    }
    cudaMemset(set, 0, sizeof(u32) * n);
    syncCheck();

    for (u32 j = 0; j < repeat; j += 1) {
      auto dt = time_func([&] { t->lookup(key, set, n); });
      sum += dt;
      sum2 += dt * dt;

      syncCheck();
    }

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16u%-16.4f%-16.4f%-16.4f\n", i, (n / 10e6) / (mean / 10e3), mean,
           stddev);
  }

  delete t;
  cudaFree(key);
}
