#include <cstdio>

#include "cuda.h"
#include "curand.h"

#include "DeviceTable.h"
#include "Types.h"
#include "xxHash.h"

__global__ void hello() {
  printf("blockIdx.x=%d/%d blocks, threadIdx.x=%d/%d threads\n", blockIdx.x,
         gridDim.x, threadIdx.x, blockDim.x);
}

__global__ void genRandArray(u32 *array, u32 n) {
    u32 id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id < n) {
        u32 acc = xxhash((u32)(usize)(array), n);
        acc = xxhash(acc, threadIdx.x);
        acc = xxhash(acc, blockIdx.x);
        acc = xxhash(acc, blockDim.x);
        array[id] = acc;
    }
}

__global__ void printArray(u32 *array, u32 n) {
    u32 id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        printf("%x\n", array[id]);
    }
}

void wrapper() {
    u32 *array = nullptr;
    cudaMalloc(&array, sizeof(u32) * 1024);
    genRandArray<<<1, 1024>>>(array, 1024);
    printArray<<<1, 1024>>>(array, 1024);
}
