#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;

#define DIM 16

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
#define M 16777216
#define N 16
#define K 16

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

// uncomment this line to print out the arrays
//#define DEBUG 1

__global__ void single_kernel_development_version(float* particles, float* centroids, int32_t* clusterAssignmentVector){

    __shared__ float                  particlesTile[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half               particlesTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        particlesSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ float                  centroidsTile[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half               centroidsTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        centroidsSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half     centroidsTileHalfTransposed[BLOCK_WIDTH][BLOCK_HEIGHT];

    __shared__ half                      sumSquares[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half                      resultTile[BLOCK_HEIGHT][BLOCK_WIDTH];

    // these variables are a waste of registers but make the code a little bit cleaner
    int globalThreadIndex_X = threadIdx.x;
    int globalThreadIndex_Y = threadIdx.y + blockIdx.y*blockDim.y + blockIdx.x*blockDim.y*gridDim.y;

    // load up particlesTile from global memory (use float4s to guarantee alignment + vectorise with LDS128?)
    particlesTile[threadIdx.y][threadIdx.x] = particles[globalThreadIndex_X + K * globalThreadIndex_Y];

    // convert to half precision for use in wmma operation (fuse this into above line?)
    particlesTileHalf[threadIdx.y][threadIdx.x] = __float2half(particlesTile[threadIdx.y][threadIdx.x]);

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
    centroidsTile[threadIdx.y][threadIdx.x] = centroids[threadIdx.x + N * threadIdx.y];

    // I think this is the first __syncthreads() needed because we have inter-warp transfer for the transpose
    __syncthreads();

    // downcast centroidsTile into centroidsTileHalf (not sure but could be bank conflicts here!)
    centroidsTileHalf[threadIdx.y][threadIdx.x] = __float2half(centroidsTile[threadIdx.y][threadIdx.x]);
    __syncthreads();

    // transpose centroidsTileHalf into centroidsTileHalfTransposed (not sure but could be bank conflicts here!)
    centroidsTileHalfTransposed[threadIdx.y][threadIdx.x] = centroidsTileHalf[threadIdx.x][threadIdx.y];

    // make sure the above line has completed (I think this might be an unnecessary __syncthreads()? need to think more)
    __syncthreads();

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

    __syncthreads();

#ifdef DEBUG
    // print the arrays to check them

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\nparticlesTile:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", particlesTile[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\nparticlesTileHalf:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)particlesTileHalf[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\nparticlesSquaredTileHalf:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)particlesSquaredTileHalf[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\ncentroidsTile:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", centroidsTile[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\ncentroidsTileHalfTransposed:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)centroidsTileHalfTransposed[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\ncentroidsSquaredTileHalf:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)centroidsSquaredTileHalf[i][j]);
            }
            printf("\n");
        }
    }

    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\nsumSquares:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)sumSquares[i][j]);
            }
            printf("\n");
        }
    }
#endif

    // instead of using the transposing centroids could I just declare it as column major when its actually row major?
    wmma::fragment<wmma::matrix_a,    16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, (half*)particlesTileHalf,            16);
    wmma::load_matrix_sync(b_frag, (half*)centroidsTileHalfTransposed,  16);
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

#ifdef DEBUG
    if (globalThreadIndex_X == 0 && globalThreadIndex_Y == 0){
        printf("\nresultTile:\n");
        for(int i = 0; i < BLOCK_HEIGHT; i++){
            for(int j = 0; j < BLOCK_WIDTH; j++){
                printf("%f ", (float)resultTile[i][j]);
            }
            printf("\n");
        }
    }
#endif

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
        clusterAssignmentVector[globalThreadIndex_Y] = lowestDistanceIndex;
    }
}

__global__ void single_kernel_optimised(float* particles, float* centroids, int32_t* clusterAssignmentVector){
    __shared__ half               particlesTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        particlesSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ half               centroidsTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half        centroidsSquaredTileHalf[BLOCK_HEIGHT][BLOCK_WIDTH];

    __shared__ half                      sumSquares[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ half                      resultTile[BLOCK_HEIGHT][BLOCK_WIDTH];

    // these variables are a waste of registers but make the code a little bit cleaner
    int globalThreadIndex_X = threadIdx.x;
    int globalThreadIndex_Y = threadIdx.y + blockIdx.y*blockDim.y + blockIdx.x*blockDim.y*gridDim.y;

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
        clusterAssignmentVector[globalThreadIndex_Y] = lowestDistanceIndex;
    }

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
    float *particles;
    cudaMalloc(&particles, size_p);

    int32_t *output;
    cudaMalloc(&output, size_out);
    int32_t *output_wmma;
    cudaMalloc(&output_wmma, size_out);

    cudaMemcpy(particles, host_particles, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids, host_centroids, size_c, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    float elapsedTime;

/********************************************************************************/
// MY VERSION

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockDim = dim3(BLOCK_HEIGHT, BLOCK_WIDTH);

    int numBlocksX = 32;
    int numBlocksY = (int)ceil((double)M / (double)BLOCK_HEIGHT / (double)numBlocksX);
      
    dim3 gridDim = dim3(numBlocksX, numBlocksY);

    int nRepeats = 10;

    cudaEventRecord(start, 0);
    for (int repeat = 0; repeat < nRepeats; repeat++) single_kernel_development_version<<<gridDim, blockDim>>>(particles, centroids, output_wmma);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    //printf("\nTime for wmma version: %f ms\n", elapsedTime/(double)nRepeats);
    double throughput_my_version = (double)nRepeats*size_p_gb / (elapsedTime / 1000);
    //printf("Input data / execution time: %f GB/s\n", throughput_my_version);

    cudaEventRecord(start, 0);
    for (int repeat = 0; repeat < nRepeats; repeat++) single_kernel_optimised<<<gridDim, blockDim>>>(particles, centroids, output_wmma);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nTime for single kernel wmma optimised version: %f ms\n", elapsedTime/(double)nRepeats);
    throughput_my_version = (double)nRepeats*size_p_gb / (elapsedTime / 1000);
    printf("Input data / execution time: %f GB/s", throughput_my_version);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

/********************************************************************************/
// OLD VERSION


    int numThreads = 256;
    int numBlocks = (nParticles + numThreads - 1) / numThreads;

    for (int repeat = 0; repeat < nRepeats; repeat++) assign_labels_very_slowly<<<numBlocks, numThreads>>>(centroids, particles, output, dimensions, nParticles, nCentroids);

    int32_t *host_output = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output, output, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++) {
        int32_t particle = host_output[i];
        if (particle < 0 || particle >= nCentroids) {
            std::cout << "bad" << std::endl;
            return 1;
        }
        counts[particle]++;
    }

    int32_t *host_output_wmma = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output_wmma, output_wmma, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts_wmma = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++) {
        int32_t particle = host_output_wmma[i];
        if (particle < 0 || particle >= nCentroids) {
            std::cout << "bad" << std::endl;
            return 1;
        }
        counts_wmma[particle]++;
    }

    std::cout << std::endl;
    for (int32_t i = 0; i < nCentroids; i++) {
        //std::cout << counts[i] << "\t from naive and " << counts_wmma[i] << "\t from wmma to centroid " << i << std::endl;
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