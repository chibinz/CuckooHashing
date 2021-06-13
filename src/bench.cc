#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"

#include "bench.h"
#include "device.h"
#include "multi.h"
#include "stash.h"
#include "types.h"

using Table = StashTable;

constexpr u32 repeat = 16;

using namespace std::chrono;
using Clock = high_resolution_clock;
inline double time_func(std::function<void()> f) {
  float duration;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventRecord(start, 0);

  // auto start = Clock::now();

  f();

  // auto end = Clock::now();
  // auto nano = duration_cast<nanoseconds>(end - start).count();

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

  auto pass = true;

  for (u32 i = 0; i < entry; i += 1) {
    if (set[i] != 1) {
      pass = false;
      printf("%u\n", i);
    }
  }

  cudaFree(key);
  delete t;

  assert(pass && "Failed correctness testing!");
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
      t->threshold = 4 * bit_width(n);

      auto dt = time_func([&] { t->insert(key, n); });
      sum += dt;
      sum2 += dt * dt;

      delete t;
      syncCheck();
    }

    cudaFree(key);

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16u%-16.4f%-16.4f%-16.4f\n", s, (n / 1e6) / (mean / 1e3), mean,
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
  t->threshold = 4 * bit_width(n);
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
    printf("%-16u%-16.4f%-16.4f%-16.4f\n", i, (n / 1e6) / (mean / 1e3), mean,
           stddev);
  }

  delete t;
  cudaFree(key);
}

void stress() {
  printf("%s\n", "Stress test");
  printf("%-16s%-16s%-16s%-16s%-16s\n", "s", "Insertion/Mops", "Collision/%",
         "Mean/ms", "StdDev/ms");

  double scale[] = {2.0, 1.9, 1.8, 1.7,  1.6,  1.5, 1.4,
                    1.3, 1.2, 1.1, 1.05, 1.02, 1.01};

  u32 *key, n = 1 << 24;
  cudaMalloc(&key, sizeof(u32) * n);
  randomizeDevice(key, n);
  syncCheck();

  for (auto s : scale) {
    u32 capacity = n * s, collision = 0;
    double sum = 0.0, sum2 = 0.0;

    for (u32 j = 0; j < repeat; j += 1) {
      auto t = new Table(capacity);
      t->threshold = bit_width(capacity);

      auto dt = time_func([&] { t->insert(key, n); });
      sum += dt;
      sum2 += dt * dt;
      collision += t->collision;

      delete t;
      syncCheck();
    }

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16.2f%-16.4f%-16.2f%-16.4f%-16.4f\n", s, (n / 1e6) / (mean / 1e3),
           (collision * 1e2) / (repeat * n), mean, stddev);
  }

  cudaFree(key);
}

void evict() {
  printf("%s\n", "Eviction bound test");
  printf("%-16s%-16s%-16s%-16s%-16s\n", "e", "Insertion/Mops", "Collision/%",
         "Mean/ms", "StdDev/ms");

  u32 *key, n = 1 << 24;
  cudaMalloc(&key, sizeof(u32) * n);
  randomizeDevice(key, n);
  syncCheck();

  double evict[] = {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                    0.9, 1.0, 2.0, 4.0, 8.0, 16.0};

  for (auto e : evict) {
    u32 capacity = n * 1.4, collision = 0;
    double sum = 0.0, sum2 = 0.0;

    for (u32 j = 0; j < repeat; j += 1) {
      auto t = new Table(capacity);
      t->threshold = e * bit_width(n);
      syncCheck();

      auto dt = time_func([&] { t->insert(key, n); });
      sum += dt;
      sum2 += dt * dt;
      collision += t->collision;

      delete t;
      syncCheck();
    }

    double mean = sum / (double)(repeat);
    double stddev = sqrt((sum2 / (double)(repeat)) - mean * mean);
    printf("%-16.1f%-16.4f%-16.2f%-16.4f%-16.4f\n", e, (n / 1e6) / (mean / 1e3),
           (collision * 1e2) / (repeat * n), mean, stddev);
  }

  cudaFree(key);
}
