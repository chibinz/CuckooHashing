#pragma once

#include "cuda.h"

#include "Types.h"

/// xxHash for a single 32-bit integer
__host__ __device__ u32 xxhash(u32 seed, u32 v);
