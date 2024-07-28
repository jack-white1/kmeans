#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;

#define DIM 16

//#define DEBUG 1

__global__ void assign_labels_very_slowly(float *centroids, float *particles, int32_t *output, int32_t dimensions, int32_t nParticles, int32_t nCentroids)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nParticles)
    {
        float lowestDist = 0;
        int32_t closestCentroidIdx = -1;
        for (int32_t j = 0; j < nCentroids; j++)
        {
            float dist = 0;
            for (int32_t k = 0; k < dimensions; k++)
            {
                float d = centroids[j * dimensions + k] - particles[idx * dimensions + k];
                dist += d * d;
            }
            if (dist < lowestDist || closestCentroidIdx == -1)
            {
                closestCentroidIdx = j;
                lowestDist = dist;
            }
        }
        output[idx] = closestCentroidIdx;
    }
}


// Matrix dimensions
// M is number of particles
#define M 52428800
#define N 16
#define K 16

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

// uncomment this line to print out the arrays

__global__ void single_kernel_optimised(float* particles, float* centroids, int32_t* centroidAssignmentVector, curandState *state, int numberOfParticles){
    __shared__ half               particlesTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        particlesSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ half               centroidsTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        centroidsSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ half                      sumSquares[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half                      resultTile[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ unsigned int              randomInt;
    // calculate which data we are taking
    if (threadIdx.x == 0) {
        curandState localState = state[blockIdx.y];
        unsigned int x = curand(&localState) % (numberOfParticles/BLOCK_HEIGHT);
        randomInt = x;
        state[blockIdx.y] = localState;
    }
    __syncthreads();

    int virtualBlockIdxY = randomInt;
    int globalThreadIndex_X = threadIdx.x;

    // get a RANDOM CHUNK OF DATA FOR THIS BLOCK (needed for mini-batch k-means)
    int globalThreadIndex_Y = threadIdx.y + virtualBlockIdxY*blockDim.y;

    // load up particlesTile from global memory (use float4s to guarantee alignment + vectorise with LDS128?)
    particlesTileHalf[threadIdx.y][threadIdx.x] =  __float2half(particles[globalThreadIndex_X + K * globalThreadIndex_Y]);

    // calculate particlesSquaredTileHalf with a pointwise ^2 followed by sum reduction across the row
    half myVal = particlesTileHalf[threadIdx.y][threadIdx.x];
    half myValSquared = myVal * myVal;
    half myValSquaredSum = myValSquared;

    for (int i = 1; i < 16; i++){
        myValSquaredSum += __shfl_down_sync(0xffffffff, myValSquared, i);
    }

    // write out the row (vectorise this by doing BLOCK_WIDTH/2 __half2 writes instead of BLOCK_WIDTH half writes?)
    if (threadIdx.x == 0){
        for (int i = 0; i < BLOCK_WIDTH; i++){
            particlesSquaredTileHalf[threadIdx.y][i] = myValSquaredSum;
        }
    }

    // load up centroidsTile (do this from __constant__ memory when centroids is small? use float4s to guarantee alignment + vectorise?)
    centroidsTileHalf[threadIdx.y][threadIdx.x] = __float2half(centroids[threadIdx.x + N * threadIdx.y]);

    // calculate centroidsSquaredTileHalfTransposed with a pointwise ^2 followed by sum reduction across the row
    myVal = centroidsTileHalf[threadIdx.y][threadIdx.x];
    myValSquared = myVal * myVal;
    myValSquaredSum = myValSquared;

    for (int i = 1; i < 16; i++){
        myValSquaredSum += __shfl_down_sync(0xffffffff, myValSquared, i);
    }

    // write out the row (vectorise this by doing BLOCK_WIDTH/2 __half2 writes instead of BLOCK_WIDTH half writes?)
    if (threadIdx.x == 0){
        for (int i = 0; i < BLOCK_WIDTH; i++){
            centroidsSquaredTileHalf[i][threadIdx.y] = myValSquaredSum; 
        }
    }

    __syncthreads();

    // combine the squared matrices into sumSquares
    sumSquares[threadIdx.y][threadIdx.x] = (centroidsSquaredTileHalf[threadIdx.y][threadIdx.x] + particlesSquaredTileHalf[threadIdx.y][threadIdx.x])/(half)-2.0;

    // instead of using the transposing centroids could I just declare it as column major when its actually row major?
    wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, (half*)particlesTileHalf,            16);
    wmma::load_matrix_sync(b_frag, (half*)centroidsTileHalf,  16);
    wmma::load_matrix_sync(c_frag, (half*)sumSquares,                   16, wmma::mem_row_major);

    // the result I want is sumsquares - 2.0*particles*centroids_t
    // however the wmma function will give me particles * centroids_t
    // so I need to do particles*centroids_t - 0.5 * sumsquares
    // and then post multiply pointwise by -2.0

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync((half*)resultTile, c_frag, 16, wmma::mem_row_major);
    __syncthreads();

    // postmultiply result by -2.0f
    resultTile[threadIdx.y][threadIdx.x] = (half)-2.0f * resultTile[threadIdx.y][threadIdx.x];

    __syncthreads();

    int32_t lowestDistanceIndex;

    if (threadIdx.x == 0){
        half startingValue = resultTile[threadIdx.y][threadIdx.x];
        lowestDistanceIndex = 0;
        for (int32_t i = 1; i < BLOCK_WIDTH; i++){
            if (resultTile[threadIdx.y][i] < startingValue){
                startingValue = resultTile[threadIdx.y][i];
                lowestDistanceIndex = i;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0){
        centroidAssignmentVector[globalThreadIndex_Y] = lowestDistanceIndex;
    }

}


// update centroid centers
__global__ void updateCentroids(float* particles, int32_t* centroidAssignmentVector, float* newCentroids, int32_t* centroidCounts){
    int globalThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int32_t centroidIndex;

    centroidIndex = centroidAssignmentVector[globalThreadIdx];

    // it will be -1 if the particle has not been not assigned to any centroid
    if (centroidIndex >= 0){
        for (int i = 0; i < K; i++){
            atomicAdd(&(newCentroids[centroidIndex*K+i]),particles[globalThreadIdx*K + i]);
        }
        atomicAdd(&(centroidCounts[centroidIndex]),1);
    }
    
}

//only to be called with blockdim = 32x32
__global__ void divideCentroidsByCounts(float* newCentroids, float* oldCentroids, int32_t* centroidCounts){
    //printf("ThreadIdx.x: %d, threadIdx.y: %d, blockIdx.x: %d, blockIdx.y: %d, dividing %.3f by %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, newCentroids[threadIdx.y*K+threadIdx.x], centroidCounts[threadIdx.y]);
    newCentroids[threadIdx.y*K+threadIdx.x] /= (float)centroidCounts[threadIdx.y];
    //atomicAdd(&oldCentroids[threadIdx.y*K+threadIdx.x],-newCentroids[threadIdx.y*K+threadIdx.x]);
}

__global__ void setup_PRNG_states(curandState *state, unsigned long long seed)
{
    int id = blockIdx.x;
    curand_init(seed, id, 0, &state[id]);
}


int main() {
    // print the model of GPU we are using
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU model: " << prop.name << std::endl;

    int64_t nParticles = M;
    int32_t dimensions = DIM;
    int32_t nCentroids = DIM;

    std::cout << "Particles: " << nParticles << ", Centroids: " << nCentroids << ", Dimensions: " << dimensions << std::endl;

    size_t size_p = nParticles * dimensions * sizeof(float);
    size_t size_c = nCentroids * dimensions * sizeof(float);
    double size_p_gb = (double)size_p / 1000000000;

    size_t size_out = nParticles * sizeof(int32_t);

    float *host_centroids = (float *)malloc(size_c);
    float *host_particles = (float *)malloc(size_p);

    srand(1337);

    for (int i = 0; i < nParticles; i++) {
        for (int j = 0; j < dimensions; j++) {
            host_particles[i * dimensions + j] = (float)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < nCentroids; i++) {
        for (int j = 0; j < dimensions; j++) {
            host_centroids[i * dimensions + j] = (float)rand() / RAND_MAX;
        }
    }

    float *centroids;
    cudaMalloc(&centroids, size_c);
    float *newCentroids;
    cudaMalloc(&newCentroids, size_c);
    cudaMemset(newCentroids,0,size_c);
    int32_t *centroidCounts;
    cudaMalloc(&centroidCounts, nCentroids * sizeof(int32_t));
    cudaMemset(centroidCounts,0,nCentroids * sizeof(int32_t));



    float *particles;
    cudaMalloc(&particles, size_p);

    int32_t *output;
    cudaMalloc(&output, size_out);
    int32_t *output_wmma;
    cudaMalloc(&output_wmma, size_out);
    cudaMemset(output_wmma, -1, size_out);

    cudaMemcpy(particles, host_particles, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids, host_centroids, size_c, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    float elapsedTime;

    dim3 blockDim = dim3(BLOCK_HEIGHT, BLOCK_WIDTH);

    int numBlocksX = 1;
    //const int numBlocksY = (int)ceil((double)M / (double)BLOCK_HEIGHT / (double)numBlocksX);

    double sample_fraction = 0.01;   // fraction of data that will be processed by mini batch

    // divide twice by BLOCK_HEIGHT is intentional
    const int numBlocksY = (int)ceil((double)M*sample_fraction / (double)BLOCK_HEIGHT / (double)BLOCK_HEIGHT);

    if (numBlocksY > 65535) {
        printf("Too many blocks in y direction, reduce sample fraction\n");
        return 1;
    }
    
    dim3 gridDim = dim3(numBlocksX, numBlocksY);

    curandState *devStates;
    unsigned int *devRandomInts;

    /* Allocate space for PRNG states and random integers on the device */
    cudaMalloc((void **)&devStates, numBlocksY * sizeof(curandState));
    cudaMalloc((void **)&devRandomInts, numBlocksY * sizeof(unsigned int));


    setup_PRNG_states<<<numBlocksY, 1>>>(devStates, time(NULL));

/********************************************************************************/
// MY VERSION
    int numberOfKernelTimingRepeats = 1;

    int numberOfIterations = 5;

    for (int iteration = 0; iteration < numberOfIterations; iteration++) {
    
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (int repeat = 0; repeat < numberOfKernelTimingRepeats; repeat++) single_kernel_optimised<<<gridDim, blockDim>>>(particles, centroids, output_wmma, devStates, M);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("\nTime for mini batch (%.1lf %% of dataset) k-means iteration : %.3f ms\n", sample_fraction*100, elapsedTime/(double)numberOfKernelTimingRepeats);
        double throughput_my_version = (double)numberOfKernelTimingRepeats*size_p_gb / (elapsedTime / 1000);
        //printf("Input data / execution time: %.3f GB/s\n", throughput_my_version);

        cudaEventRecord(start, 0);
        updateCentroids<<<nParticles/512, 512>>>(particles, output_wmma, newCentroids, centroidCounts);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Time update centroids: %.3f ms\n", elapsedTime);

        cudaEventRecord(start, 0);
        divideCentroidsByCounts<<<1, dim3(16,16)>>>(newCentroids, centroids, centroidCounts);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Time divide centroids: %.3f ms\n", elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);


#ifdef DEBUG
        float *host_new_centroids = (float *)malloc(size_c);
        cudaMemcpy(host_new_centroids, newCentroids, size_c, cudaMemcpyDeviceToHost);

        printf("\nNew centroids:\n");
        for (int i = 0; i < nCentroids; i++) {
            for (int j = 0; j < dimensions; j++) {
                printf("%.2f ", host_new_centroids[i * dimensions + j]);
            }
            printf("\n");
        }

        float *host_old_centroids = (float *)malloc(size_c);
        cudaMemcpy(host_old_centroids, centroids, size_c, cudaMemcpyDeviceToHost);

        printf("\nCentroids before update:\n");
        for (int i = 0; i < nCentroids; i++) {
            for (int j = 0; j < dimensions; j++) {
                printf("%.2f ", host_old_centroids[i * dimensions + j]);
            }
            printf("\n");
        }
#endif


        // swap old centroids with new centroids
        float *temp = centroids;
        centroids = newCentroids;
        newCentroids = temp;
        cudaMemset(newCentroids,0,size_c);
        cudaMemset(centroidCounts,0,nCentroids * sizeof(int32_t));

#ifdef DEBUG
        //copy old centroids and new centroids back to host and print them
        cudaMemcpy(host_old_centroids, centroids, size_c, cudaMemcpyDeviceToHost);

        printf("\nCentroids after update:\n");
        for (int i = 0; i < nCentroids; i++) {
            for (int j = 0; j < dimensions; j++) {
                printf("%.2f ", host_old_centroids[i * dimensions + j]);
            }
            printf("\n");
        }

        cudaMemcpy(host_new_centroids, newCentroids, size_c, cudaMemcpyDeviceToHost);

        printf("\nNew centroids:\n");
        for (int i = 0; i < nCentroids; i++) {
            for (int j = 0; j < dimensions; j++) {
                printf("%.2f ", host_new_centroids[i * dimensions + j]);
            }
            printf("\n");
        }
#endif
}

    

/********************************************************************************/
// OLD VERSION


    int numThreads = 256;
    int numBlocks = (nParticles + numThreads - 1) / numThreads;

    for (int repeat = 0; repeat < numberOfKernelTimingRepeats; repeat++) assign_labels_very_slowly<<<numBlocks, numThreads>>>(centroids, particles, output, dimensions, nParticles, nCentroids);

    int32_t *host_output = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output, output, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++) {
        int32_t particle = host_output[i];
        if (particle < 0 || particle >= nCentroids) {
            std::cout << "\nbad output slow" << std::endl;
            return 1;
        }
        counts[particle]++;
    }

    int32_t *host_output_wmma = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output_wmma, output_wmma, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts_wmma = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++) {
        int32_t particle = host_output_wmma[i];
        if (particle != -1){
            if (particle >= nCentroids) {
                std::cout << "\nbad output wmma" << std::endl;
                std::cout << particle << std::endl;
                return 1;
            }
            counts_wmma[particle]++;
        }
    }


    std::cout << std::endl;
    printf("Results with mini-batch @ %.1lf%% of the overall dataset:\n", sample_fraction*100);
    for (int32_t i = 0; i < nCentroids; i++) {
        std::cout << counts[i] << "\t from full gold standard and " << counts_wmma[i] << "\t from mini-batch to centroid " << i << std::endl;
    }

    cudaError_t err;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }

    cudaFree(centroids);
    cudaFree(particles);
    cudaFree(output);
    cudaFree(output_wmma);
    free(host_centroids);
    free(host_particles);
    free(host_output);
    free(counts);
    free(host_output_wmma);
    free(counts_wmma);

    return 0;
}