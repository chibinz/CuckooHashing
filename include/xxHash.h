#pragma once

#include "cuda.h"

#include "Types.h"

#define __dual__ __host__ __device__

/// xxHash for a single 32-bit integer
__dual__ u32 xxhash(u32 seed, u32 v);
