#pragma once

#include <cstdlib>

#include "cuda.h"

#include "Types.h"
#include "xxHash.h"

constexpr u32 empty = (u32)(-1);

namespace {

void swap(u32 *a, u32 *b) {
  u32 temp = *a;
  *a = *b;
  *b = temp;
}

void randomize(u32 *array, u32 n) {
  for (usize i = 0; i < n; i += 1) {
    array[i] = xxhash(i, array[i]);
  }
}

void syncCheck() {
  cudaDeviceSynchronize();
  auto err = cudaGetLastError(); // Get error code
  if (err != cudaSuccess) {
    printf("Error: %s!\n", cudaGetErrorString(err));
    exit(-1);
  }
}

u32 bit_width(u32 x) {
  u32 w = 0;

  while (x > 0) {
    w += 1;
    x >>= 1;
  }

  return w;
}

} // namespace
