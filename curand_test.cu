#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void generate_kernel(curandState *state, int M, unsigned int *randomInts)
{
    int blockId = blockIdx.x;
    if (threadIdx.x == 0) {
        curandState localState = state[blockId];
        unsigned int x = curand(&localState) % M;
        randomInts[blockId] = x;
        state[blockId] = localState;
    }
    __syncthreads();
    printf("Block %d, Thread %d: %u\n", blockId, threadIdx.x, randomInts[blockId]);
}

__global__ void setup_kernel(curandState *state, unsigned long long seed)
{
    int id = blockIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

int main(int argc, char *argv[])
{
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    int M = 16777216; // Default value for M

    if (argc >= 2) {
        M = atoi(argv[1]);
    }

    curandState *devStates;
    unsigned int *devRandomInts;

    /* Allocate space for PRNG states and random integers on the device */
    CUDA_CALL(cudaMalloc((void **)&devStates, blockCount * sizeof(curandState)));
    CUDA_CALL(cudaMalloc((void **)&devRandomInts, blockCount * sizeof(unsigned int)));

    /* Setup PRNG states */
    setup_kernel<<<blockCount, 1>>>(devStates, time(NULL));

    /* Generate and print random numbers */
    generate_kernel<<<blockCount, threadsPerBlock>>>(devStates, M, devRandomInts);

    /* Cleanup */
    CUDA_CALL(cudaFree(devStates));
    CUDA_CALL(cudaFree(devRandomInts));
    CUDA_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}
