# CuckooHashing
Concurrent cuckoo hash table implemented in CUDA

## Build & Run
```
# Install dependencies
sudo apt install meson ninja-build nvidia-cuda-toolkit

# Build
meson setup build
meson compile -C build

# Run
build/chash
```





## Experiments
### Spec

Entry  | |
------ | -----
CPU    | x
GPU    | x
Memory | x
Operating System | x
CUDA Version | 9.1

### Correctness testing
We verify the correctness of our implementation through inserting 512 elements and performing lookup on the same elements. If all elements are marked as inserted, we conclude that our implementation is correct. Different seeds are used and many trials are performed to obtain reliable conclusion.

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

The increase in performance for the naive and multilevel table could be explained by reduction in divergence. While finding a key early means exiting the loop early, other threads in the same warp may not take the same path. This is especially true when i = 0, where keys could be located at all 3 sub tables. The warp scheduler may have to execute every iteration of the for loop twice depending on the distribution of keys. When i = 10, all key lookup fails and all threads perfectly synchronize, reducing divergence to 0.

On the other hand, trend is inverse for the stash table. This is due to the extra time needed to linearly probe the stash when lookup fails in the main table. Since the stash is very small, performance only degrades slightly.


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

The result of this experiment shows that using a stash to cope with collisions is indeed a very effective technique.

### 4. Eviction bound test
> Using n = 2^24 random keys and a hash table of size 1.4n, xperiment with different bounds on the maximum length of an eviction chain before restarting. Which bound gives the best running time for constructing the hash table? Note however you are not required to find the optimal bound.

e    | Naive  | Multilevel | Stash
---- | ------ | ---------- | -----
0.2  | N/A    | N/A        | 233.1928
0.3  | N/A    | 66.2509    | 221.4478
0.4  | N/A    | 61.8062    | 216.2449
0.5  | N/A    | 65.6564    | 217.5787
0.6  | N/A    | 80.5926    | 218.5072
0.7  | N/A    | 76.2219    | 218.3521
0.8  | N/A    | 76.6323    | 220.7911
0.9  | N/A    | 85.7437    | 222.7855
1.0  | N/A    | 82.7799    | 225.0036
2.0  | 108.61 | 91.4688    | 230.3316
4.0  | 176.50 | 91.9897    | 230.4251
8.0  | 176.49 | 91.9756    | 230.3617
16.0 | 176.58 | 91.9972    | 230.3030

## Optimizations

### Choice of Hash Function
Initially, we follow Alcantera's advice using the hash function with formula [(c0 + c1 * k) mod 1900813], where c0, c1 are 2 constants generated at initialization for each subtable. However experiment 1 shows that this is not an ideal choice. With capacity set to 2 ^ 25 and load factor about 0.3, the straightforward implementation of cuckoo hashing rehashes indefinitely and never terminates.

After sometime researching online comparing the merits and demerits of several hash functions, we settled on `xxHash32` as our hash function. Benchmarks performed by Jarzynski et al. suggests that `xxHash32` strikes a good balance between run time performance and hash table collision rate. Switching to `xxHash32` proved to be wise choice with substantial decrease in collision count and negligible runtime overhead. The maximum load factor that the straightforward table was able is contain is 0.5, though careful selection of random seed is needed. Another benefit of `xxHash32` is that it can be seeded and only require a single u32 as seed. This allows to generate different hash function for each subtable very efficiently.

### Random Number Generation
At the beginning, we use the same hash function used in our hash table `xxHash32` to generate random integers. We also tried cuda's default random number generator, the `curand` library. Unfortunately, it only supports generating floating point numbers and not suitable for our use case. To be scientific and firm in our reportings, our final implementation uses the `rand()` function from `<stdlib.h>`. Random integers are generated and stored in host memory and later `memcpy`ed to device memory. Duplicates are removed using `std::set`.

### Straightforward Table
We first implement straighforward version of Cuckoo hashing. While cuckoo hashing is a recursive process, device functions usually does not favor function calls. Thus the eviction chain is implemented as a for loop with maximum loop count set to eviction bound. If an empty slot is found, we exit the loop early. If no empty slot is found and the maximum iteration is reached, we increment `collision` flagging to host that a rehash is needed. In practice, this rarely happens when load factor is less than 0.5.

Note unlike conventional hash tables, accesses to table slots are made simultaneously by multiple GPU threads. To cope with concurrency, we use `atomicExch` to atomically swap key to be hashed and the target table slot. We terminate the loop if an `empty` value is retrieved, i.e., the key is successfully hashed, otherwise continue on the eviction chain.

### Multilevel Table
Drawing  from what we've learnt in class, frequent uncoalesced memory access to global memory is often times a performance kill, let alone all accesses must be made atomically. Following this insight, we try to implement a multilevel hahs table where the keys are first hashed into buckets in global memory, and cuckoo hashing is performed on individual buckets in shared memory. If any of the first level buckets is found out to be full, we generate a new hash function, perform a rehash and try to distribute the keys into the hash table more evenly. After the keys are distributed into buckets, we perform cuckoo hashing on individual buckets. If unfortunate choice of hash function is made, we only rehash the conflicting buckets.

While the insertion time alone is smaller than the straightforward table, combining the time to bucket input puts the multilevel table in disadvantage. Quite contrary to our initial expectation. After extensive research online we've found that global atomics are optimized from the Kepler and Maxwell architecture and then on. Performing atomics on shared memory could actually hurt performance. We regard this as an real life example of how architectural improvement in hardware could turn a turn software optimization into deoptimization. Our result is consistent with more recent publications of xxx and xxx.

There are still some advantage of using a multilevel table though. The naive table simply throws the towel when load factor approaches 0.5. While the multilevel table persists till 0.85, or when capacity = 1.2 n. It is also less sensitive to the eviction bound size, reducing the time tuning this parameter.

### Stash Table
A third approach that we took comes from the 2 observations:
1. When load factor is small, collisions happen rather infrequently. A full table rehash in such case is not worth extra overhead.
2. When load factor is large, collisions are inevitable. Whether or not the hash table will able to contain all keys is a matter of probability and such probability decreases sharply after load factor reaches 0.5.
Following these observations, we conclude that it might be worthwhile to set aside a small stash to store conflicting keys. This complements the good performance of the straightforward table when the load is small, and helps us advance through even the ill-formulated capacity = 1.01n, achieving the best of both world. Experimental results show that the stash table has the best overall performance.


## Credit
1. [Alcantara, Dan A., et al. "Real-time parallel hashing on the GPU."](https://hal.inria.fr/inria-00624777/document)
2. [Cassee, Nathan, and Anton Wijs. "Analysing the performance of GPU hash tables for state space exploration."](https://arxiv.org/pdf/1712.09494)
3. [Jarzynski, Mark, and Marc Olano. "Hash Functions for GPU Rendering."](https://mdsoar.org/bitstream/handle/11603/20126/paper.pdf?sequence=6&isAllowed=y)
4. [cudpp/cudpp](https://github.com/cudpp/cudpp)

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
