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

using Table = DeviceTable;
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
