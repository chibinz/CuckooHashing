#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"

#include "bench.h"
#include "device.h"
#include "multi.h"
#include "types.h"

using namespace std::chrono;

using Table = MultilevelTable;
using Clock = high_resolution_clock;

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

void insertion() {
  printf("%-16s%-16s%-16s\n", "s", "Mean/ms", "StdDev/ms");

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
    printf("%-16u%-16.4f%-16.4f\n", s, mean, stddev);
  }
}

void lookup() {
  printf("%s\n", "Lookup");
  printf("%-16s%-16s%-16s\n", "i", "Mean/ms", "StdDev/ms");

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
    printf("%-16u%-16.4f%-16.4f\n", i, mean, stddev);
  }

  delete t;
  cudaFree(key);
}
