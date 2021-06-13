# CuckooHashing
Concurrent cuckoo hash table implemented in CUDA

## Build
```
sudo apt install meson ninja-build nvidia-cuda-toolkit
meson setup build
meson compile -C build
```

## Run
```
build/chash
```
## Overview

## Experiments
### Spec

Entry  | |
------ | -----
CPU    | x
GPU    | x
Memory | x
Operating System | x
CUDA Version | 9.1

### 1. Insertion
> Create a hash table of size 2^25 in GPU global memory, where each table entry stores a 32-bit integer. Insert a set of 2^s random integer keys into the hash table, for s = 10, 11, ..., 24.

s  | Naive | Multilevel | Stash
-- | ----- | ---------- | -----
10 | x     |            |
11 | x     |            |
12 | x     |            |
13 | x     |            |
14 | x     |            |
15 | x     |            |
16 | x     |            |
17 | x     |            |
18 | x     |            |
19 | x     |            |
21 | x     |            |
22 | x     |            |
23 | x     |            |
24 | x     |            |

### 2. Lookup
> Insert a set Sܵ of 2^24 random keys into a hash table of size 2^25, then perform lookups for the following sets of keys ܵS_0, ..., S_10. Each set ܵS_i should contain 2^24 keys, where (100 - 10i) percent of the keys are randomly chosen from S, and the remainder are random 32-bit keys. For example, ܵS_0 should contain only random keys from S, while S_5 should 50% random keys from S, and 50% completely random keys.

i  | Naive | Multilevel | Stash
-- | ----- | ---------- | -----
0  | x     |            |
1  | x     |            |
2  | x     |            |
3  | x     |            |
4  | x     |            |
5  | x     |            |
6  | x     |            |
7  | x     |            |
8  | x     |            |
9  | x     |            |
10 | x     |            |

### 3. Stress test
> Fix a set of n = 2^24 random keys, and measure the time to insert the keys into hash tables of sizes s = 1.1n, 1.2n, ..., 2n. ݊Also, measure the insertion times for hash tables of sizes 1.01n, 1.02݊n and 1.05n. Terminate the experiment if it takes too long and report the time used.

s    | Naive | Multilevel | Stash
---- | ----- | ---------- | -----
2.0  | x     |            |
1.9  | x     |            |
1.8  | x     |            |
1.7  | x     |            |
1.6  | x     |            |
1.5  | x     |            |
1.4  | x     |            |
1.3  | x     |            |
1.2  | x     |            |
1.1  | x     |            |
1.05 | x     |            |
1.02 | x     |            |
1.01 | x     |            |

### 4. Eviction bound test
> Using n = 2^24 random keys and a hash table of size 1.4n, xperiment with different bounds on the maximum length of an eviction chain before restarting. Which bound gives the best running time for constructing the hash table? Note however you are not required to find the optimal bound.

e    | Naive | Multilevel | Stash
---- | ----- | ---------- | -----
0.2  | x     |            |
0.3  | x     |            |
0.4  | x     |            |
0.5  | x     |            |
0.6  | x     |            |
0.7  | x     |            |
0.8  | x     |            |
0.9  | x     |            |
1.0  | x     |            |
2.0  | x     |            |
4.0  | x     |            |
8.0  | x     |            |
16.0 | x     |            |

## Reference
1. Timo Kehrer and Alice Miller (Eds.):Workshop on Graphs as Models 2017 (GAM ’17)EPTCS 263, 2017, pp. 1–15, doi:10.4204/EPTCS.263.1c©N.W. Cassee & A.J. WijsAnalysing the Performance of GPU Hash Tables for StateSpace Exploration
2. Efficient

## Important Concepts
### Computation Hierarchy
- Software concept
    - Thread
    - Block
    - Grid?

- Hardware concept
    - Hardware thread (execution unit?)
    - Warp (Cuda core)
    - Streaming Multiprocessor
    - GPU

Unlike in the CPU context, a thread is not a basic unit of scheduling, but a warp is. A block is usually assigned to a SM on a gpu. All warps in the same SM execute the same kernel (GPU function), but each has a separate program counter to keep track of state. A warp usually consists of 32 execution units, which has been true since the release of CUDA, and will likely stay so for the forseeable future. Number of threads in a block does not have to match the number of execution units in a single SM. The GPU block scheduler may deposit multiple blocks on a single SM. SMs on the Fermi architecture consist of 32 warps, totaling 1024 threads. Interestingly, the hard limit for the number of software threads allowed in a single block is also 1024.

## Naive table with no rehash
```
==17867== NVPROF is profiling process 17867, command: build/chash
Total number of collisions: 2586
==17867== Profiling application: build/chash
==17867== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   82.80%  53.242ms         1  53.242ms  53.242ms  53.242ms  batchedInsert(DeviceTable*, unsigned int*, unsigned int)
                   13.93%  8.9591ms         1  8.9591ms  8.9591ms  8.9591ms  batchedLookup(DeviceTable*, unsigned int*, unsigned int)
                    2.33%  1.4972ms         2  748.58us  4.3840us  1.4928ms  randomizeArray(unsigned int*, unsigned int)
                    0.94%  603.56us         1  603.56us  603.56us  603.56us  setEmpty(unsigned int*, unsigned int)
      API calls:   84.49%  469.78ms         4  117.44ms  25.227us  451.53ms  cudaMallocManaged
                   11.58%  64.364ms         7  9.1948ms  5.8930us  53.263ms  cudaDeviceSynchronize
                    2.87%  15.963ms         3  5.3211ms  31.361us  10.813ms  cudaFree
                    0.70%  3.8690ms         4  967.25us  371.99us  1.1911ms  cuDeviceTotalMem
                    0.25%  1.3956ms       376  3.7110us     251ns  150.54us  cuDeviceGetAttribute
                    0.08%  464.94us         5  92.987us  48.694us  195.42us  cudaLaunch
                    0.02%  135.99us         4  33.997us  30.175us  44.226us  cuDeviceGetName
                    0.00%  7.9450us        12     662ns     152ns  4.1640us  cudaSetupArgument
                    0.00%  4.5490us         3  1.5160us     309ns  3.7270us  cuDeviceGetCount
                    0.00%  3.5380us         7     505ns     222ns  1.0060us  cudaGetLastError
                    0.00%  3.4120us         8     426ns     256ns  1.0690us  cuDeviceGet
                    0.00%  3.2570us         5     651ns     265ns  1.5530us  cudaConfigureCall
```
