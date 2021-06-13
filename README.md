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
## Optimizations

### Choice of Hash Function
Initially, we follow Alcantara's advice using the hash function with the formula [(c0 + c1 * k) mod 1900813], where c0, c1 are 2 constants generated at initialization for each subtable. However, experiment 1 shows that this is not an ideal choice. With capacity set to 2 ^ 25 and load factor about 0.3, the straightforward implementation of cuckoo hashing rehashes indefinitely and never terminates.

After some time researching online comparing the merits and demerits of several hash functions, we settled on `xxHash32` as our hash function. Benchmarks performed by Jarzynski et al. suggests that `xxHash32` strikes a good balance between run time performance and hash table collision rate. Switching to `xxHash32` proved to be a wise choice with a substantial decrease in collision count and negligible runtime overhead. The maximum load factor that the straightforward table was able to contain is now over 0.5. Another benefit of `xxHash32` is that it can be seeded and only require a single u32 seed. This allows us to generate different hash functions for each subtable very efficiently.

### Random Number Generation
In the beginning, we use the same hash function used in our hash table `xxHash32` to generate random integers. We also tried cuda's default random number generator, the `curand` library. Unfortunately, it only supports generating floating-point numbers and not suitable for our use case. To be scientific and firm in our reportings, our final implementation uses the `rand()` function from `<stdlib.h>`. Random integers are generated and stored in host memory and later `memcpy'ed to the device memory. Duplicates are removed using `std::set`.

### Straightforward Table
We first implement a straightforward version of Cuckoo hashing. While cuckoo hashing is a recursive process, device functions usually do not favor function calls. Thus the eviction chain is implemented as a for loop with maximum loop count set to eviction bound. If an empty slot is found, we exit the loop early. If no open space is found and the maximum iteration is reached, we increment `collision` flagging to host that a rehash is needed. In practice, this rarely happens when the load factor is less than 0.5.

Note unlike conventional hash tables, accesses to table slots are made simultaneously by multiple GPU threads. We use `atomicExch` to atomically swap keys to be hashed and the target table slot to cope with concurrency. We terminate the loop if an `empty` value is retrieved, i.e., the key is successfully hashed; otherwise, we continue on the eviction chain.

### Multilevel Table
Drawing from what we've learned in class, frequent uncoalesced memory access to global memory is often a performance kill, let alone all accesses must be made atomically. Following this insight, we try to implement a multilevel hash table where the keys are first hashed into buckets in global memory. Cuckoo hashing is performed on individual buckets in shared memory. If any of the first level buckets is found out to be full, we generate a new hash function, perform a rehash and try to distribute the keys into the hash table more evenly. After the keys are distributed into buckets, we perform cuckoo hashing on individual buckets. If an unfortunate choice of a hash function is made, we only rehash the conflicting buckets.

While the insertion time alone is shorter than the straightforward table, combining the time to bucket input puts the multilevel table at a disadvantage. Quite contrary to our initial expectation. After extensive research online, we've found that global atomics are optimized from the Kepler and Maxwell architecture and then on. Performing atomics on shared memory could actually hurt performance. We regard this as a real-life example of how architectural improvement in hardware could turn software optimization into deoptimization. Our result is consistent with more recent publications of [1](https://www.sciencedirect.com/science/article/pii/B9780123859631000046) and [2](https://arxiv.org/pdf/1712.09494).

There are still some advantages of using a multilevel table, though. The naive table simply throws the towel when the load factor exceeds 0.7. While the multilevel table persists till 0.85, or when capacity = 1.2n. It is also less sensitive to the eviction bound size, reducing the time tuning this parameter.

### Stash Table
A third approach that we took comes from the 2 observations:
1. When the load factor is small, collisions happen relatively infrequently. A complete table rehash, in such a case, is not worth extra overhead.
2. When the load factor is large, collisions are inevitable. Whether or not the hash table will contain all keys is a matter of probability, and such probability decreases sharply after the load factor reaches 0.7.

Following these observations, we conclude that it might be worthwhile to set aside a small stash to store conflicting keys. This complements the excellent performance of the straightforward table when the load is small and helps us advance through even the ill-formulated capacity = 1.01n, achieving the best of both worlds. Experimental results show that the stash table has the best overall performance.

## Experiments

All performance figures are reported in Mops or million operations per second. Additional performance metrics like the total number of collisions, mean time, and standard deviation are reported for the stash table. Each experiment is repeated 32 times to gain statistical confidence. (N/A means timeout)

Config   | Value
-------- | -----
CPU      | Intel Xeon E5-2690
GPU      | Nvidia K40m (2880 CUDA cores)
Memory   | 252 GiB
Distro   | Ubuntu 18.04
Kernel   | 3.10.0
CUDA     | 9.0


### Correctness testing
We verify the correctness of our implementation by inserting 512 elements and performing lookups on the same elements. If all elements are marked as inserted, we conclude that our implementation is correct. Different seeds are used, and many trials are performed to obtain reliable conclusions.

### 1. Insertion
> Create a hash table of size 2^25 in GPU global memory, where each table entry stores a 32-bit integer. Insert a set of 2^s random integer keys into the hash table, for s = 10, 11, ..., 24.

s  | Naive    | Multilevel | Stash    | Mean/ms | StdDev/ms
-- | -------- | ---------- | -------- | ------- | ------
10 | 0.5086   | 0.0622     | 0.5308   | 1.9290  | 0.1554
11 | 1.0097   | 0.1242     | 1.0876   | 1.8830  | 0.0140
12 | 1.9824   | 0.2475     | 2.1593   | 1.8969  | 0.0157
13 | 3.9719   | 0.4878     | 4.3357   | 1.8894  | 0.0106
14 | 8.0463   | 0.8993     | 8.6290   | 1.8987  | 0.0147
15 | 15.7511  | 1.4739     | 16.8820  | 1.9410  | 0.0127
16 | 30.4085  | 2.3080     | 32.8732  | 1.9936  | 0.0145
17 | 58.0311  | 3.7947     | 62.5064  | 2.0969  | 0.0091
18 | 104.1098 | 6.8521     | 114.4582 | 2.2903  | 0.0212
19 | 176.4422 | 12.7828    | 192.5563 | 2.7228  | 0.0192
20 | 269.0715 | 22.2279    | 287.5881 | 3.6461  | 0.1272
21 | 347.9472 | 37.5608    | 381.7428 | 5.4936  | 0.0262
22 | 391.0260 | 66.5917    | 434.2630 | 9.6584  | 0.0359
23 | 374.9033 | 101.5353   | 424.4112 | 19.7653 | 0.0273
24 | 270.1257 | 118.2038   | 336.0414 | 49.9260 | 0.0452

Performance dips when s = 24 for the naive and stash table because collisions begin to happen. However, the stash table performs better because no rehash is needed.

### 2. Lookup
> Insert a set Sܵ of 2^24 random keys into a hash table of size 2^25, then perform lookups for the following sets of keys ܵS_0, ..., S_10. Each set ܵS_i should contain 2^24 keys, where (100 - 10i) percent of the keys are randomly chosen from S, and the remainder is random 32-bit keys. For example, ܵS_0 should contain only random keys from S, while S_5 should 50% random keys from S and 50% completely random keys.

i  | Naive    | Multilevel | Stash    | Mean/ms | StdDev/ms
-- | -------- | ---------- | -------- | ------- | ------
0  | 377.6520 | 252.7055   | 534.9546 | 31.3619 | 0.0567
1  | 378.4491 | 253.6125   | 523.1317 | 32.0707 | 0.0355
2  | 379.3551 | 254.0673   | 509.9108 | 32.9023 | 0.0314
3  | 379.9102 | 254.6413   | 497.3425 | 33.7337 | 0.0280
4  | 380.7537 | 255.2533   | 483.8156 | 34.6769 | 0.0349
5  | 381.4734 | 255.0833   | 470.7332 | 35.6406 | 0.0282
6  | 382.4053 | 255.7710   | 456.3187 | 36.7664 | 0.0560
7  | 382.7881 | 256.4347   | 443.1547 | 37.8586 | 0.0309
8  | 383.9326 | 256.7915   | 428.6367 | 39.1409 | 0.0411
9  | 384.3994 | 257.3034   | 413.5550 | 40.5683 | 0.0174
10 | 384.8294 | 257.6221   | 394.6965 | 42.5066 | 0.0265

The increase in performance for the naive and multilevel table could be explained by the reduction in divergence. While finding a key early means exiting the loop early, other threads in the same warp may not take the same path. This is especially true when i = 0, where keys could be located at all 3 subtables. The warp scheduler may have to execute every iteration of the for loop twice, depending on the distribution of keys. All key lookup fails when i = 10 fails, and all threads perfectly synchronize, reducing divergence to 0.

On the other hand, the trend is inverse for the stash table. This is due to the extra time needed to linearly probe the stash when the lookup fails in the main table. Since the stash is minimal, performance only degrades slightly.


### 3. Stress test
> Fix a set of n = 2^24 random keys, and measure the time to insert the keys into hash tables of sizes s = 1.1n, 1.2n, ..., 2n. ݊Also, measure the insertion times for hash tables of sizes 1.01n, 1.02݊n, and 1.05n. Terminate the experiment if it takes too long and reports the time used.

s    | Naive    | Multilevel | Stash    | Collision/% | Mean/ms  | StdDev/ms
---- | -------- | ---------- | -------- | ----------- | -------- | ------
2.0  | 270.3744 | 118.0719   | 336.2089 |  0.00       | 49.9012  | 0.0420
1.9  | 258.9772 | 116.1746   | 324.2233 |  0.00       | 51.7459  | 0.0472
1.8  | 246.4734 | 113.6895   | 310.7758 |  0.00       | 53.9850  | 0.0564
1.7  | 232.1953 | 110.5201   | 295.0688 |  0.00       | 56.8587  | 0.0712
1.6  | 216.0954 | 105.8690   | 276.5650 |  0.00       | 60.6628  | 0.0935
1.5  | 198.1926 | 100.0926   | 253.2920 |  0.00       | 66.2367  | 0.2379
1.4  | N/A      | 92.0229    | 224.8730 |  0.00       | 74.6075  | 0.1195
1.3  | N/A      | 81.3341    | 191.8155 |  0.04       | 87.4654  | 0.0855
1.2  | N/A      | 60.8193    | 156.2301 |  0.35       | 107.3879 | 0.1266
1.1  | N/A      | N/A        | 120.4658 |  2.30       | 139.2695 | 0.1096
1.05 | N/A      | N/A        | 104.7163 |  4.52       | 160.2158 | 0.1488
1.02 | N/A      | N/A        | 96.4342  |  6.22       | 173.9757 | 0.1293
1.01 | N/A      | N/A        | 93.8997  |  6.83       | 178.6717 | 0.1222

This experiment shows that using a stash to cope with collisions is indeed a very effective technique.

### 4. Eviction bound test
> Using n = 2^24 random keys and a hash table of size 1.4n, experiment with different bounds on the maximum length of an eviction chain before restarting. Which bound gives the best running time for constructing the hash table? Note, however you are not required to find the optimal bound.

e * logn | Naive    | Multilevel | Stash    | Collision/% | Mean/ms | StdDev/ms
-------- | -------- | ---------- | -------- | ----------- | ------- | ------
0.2      | N/A      | N/A        | 233.1928 | 4.29        | 71.9457 | 0.1570
0.3      | N/A      | 66.2509    | 221.4478 | 1.78        | 75.7615 | 0.2969
0.4      | N/A      | 61.8062    | 216.2449 | 0.60        | 77.5843 | 0.2400
0.5      | N/A      | 65.6564    | 217.5787 | 0.21        | 77.1087 | 0.1529
0.6      | N/A      | 80.5926    | 218.5072 | 0.08        | 76.7811 | 0.1830
0.7      | N/A      | 76.2219    | 218.3521 | 0.06        | 76.8356 | 0.1114
0.8      | N/A      | 76.6323    | 220.7911 | 0.02        | 75.9868 | 0.1059
0.9      | N/A      | 85.7437    | 222.7855 | 0.01        | 75.3066 | 0.0954
1.0      | N/A      | 82.7799    | 225.0036 | 0.00        | 74.5642 | 0.0702
2.0      | 108.6175 | 91.4688    | 230.3316 | 0.00        | 72.8394 | 0.1075
4.0      | 176.5028 | 91.9897    | 230.4251 | 0.00        | 72.8098 | 0.0923
8.0      | 176.4994 | 91.9756    | 230.3617 | 0.00        | 72.8299 | 0.0671
16.0     | 176.5834 | 91.9972    | 230.3030 | 0.00        | 72.8484 | 0.1776

The first decreasing then increasing trend in the performance of the stash table can be attributed to 2 factors:
1. The eviction bound
2. The collision rate

When the eviction bound is tiny, the number of iterations needed to complete insertion is also small. However, this comes at the cost of a higher collision rate—the collision rate decreases as the eviction bound grows. Insertion performance for all tables converges after e >= 4.0n, which is the default suggested bound. Setting e to 0.1 or smaller causes endless rehash for all tables, and thus figures are not presented.

## Credit
1. [Alcantara, Dan A., et al. "Real-time parallel hashing on the GPU."](https://hal.inria.fr/inria-00624777/document)
2. [Cassee, Nathan, and Anton Wijs. "Analysing the performance of GPU hash tables for state-space exploration."](https://arxiv.org/pdf/1712.09494)
3. [Jarzynski, Mark, and Marc Olano. "Hash Functions for GPU Rendering."](https://mdsoar.org/bitstream/handle/11603/20126/paper.pdf?sequence=6&isAllowed=y)
4. [cudpp/cudpp](https://github.com/cudpp/cudpp)
