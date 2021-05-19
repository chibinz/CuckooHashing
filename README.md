# CuckooHashing
Concurrent cuckoo hash table implemented in CUDA

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
