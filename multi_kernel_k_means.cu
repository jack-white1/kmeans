#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cublas_v2.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// using namespace nvcuda;

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


// this should be called with number of threads per block = DIM, so I guess this caps DIM at 1024
// this is just to avoid using dynamic shared memory for now
__global__ void l2_squared_norm_vec(float *particles, float *particles_squared, int32_t dimensions, int32_t nParticles)
{
    __shared__ float sdata[DIM];

    float val  = particles[blockIdx.x * blockDim.x + threadIdx.x];

    sdata[threadIdx.x] = val * val;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        particles_squared[blockIdx.x] = sdata[0];
    }
}

// 1 thread block per particle
// nCentroids threads per block
__global__ void add_particles_l2_squared_to_distances(float *particles_squared, float *distances, int32_t nParticles, int32_t nCentroids){
    int32_t idx = blockIdx.x;
    int32_t j = threadIdx.x;
    distances[idx * nCentroids + j] += particles_squared[idx];
}

// 1 thread block per particle
// nCentroids threads per block
__global__ void add_centroids_l2_squared_to_distances(float *centroids_squared, float *distances, int32_t nParticles, int32_t nCentroids){
    int32_t idx = blockIdx.x;
    int32_t j = threadIdx.x;
    distances[idx * nCentroids + j] += centroids_squared[j];
}

struct float_int
{
    float value;
    int32_t index;
};

// find min of each row of distances
// 1 thread block per particle
// nCentroids threads per block
__global__ void find_argmin(float *distances, int32_t *labels, int32_t nParticles, int32_t nCentroids){
    int particleIndex = blockIdx.x;
    int i = threadIdx.x;
    
    __shared__ float_int sdata[DIM];
    
    sdata[i].value = distances[particleIndex * nCentroids + i];
    sdata[i].index = i;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            if (sdata[threadIdx.x].value > sdata[threadIdx.x + stride].value)
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
            }
        }
    }

    if (threadIdx.x == 0)
    {
        labels[particleIndex] = sdata[0].index;
    }

}


/*************************** WMMA **************************/

// Matrix dimensions
// M is number of particles
#define M 16777216
#define N 16
#define K 16

// WMMA dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_matrix_multiply(half *A, half *B, float *C, int a, int b, int c) {
    // Calculate warp ID
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpX = warpId % (a / WMMA_M);
    int warpY = warpId / (a / WMMA_M);

    if (warpX >= (a / WMMA_M) || warpY >= (b / WMMA_N)) return;

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    nvcuda::wmma::load_matrix_sync(a_frag, A + warpX * WMMA_M * c, c);
    nvcuda::wmma::load_matrix_sync(b_frag, B + warpY * WMMA_N * c, c);

    // Perform the matrix multiplication
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    nvcuda::wmma::store_matrix_sync(C + warpX * WMMA_M * b + warpY * WMMA_N, c_frag, b, nvcuda::wmma::mem_row_major);
}

__global__ void float2half(float *input, half *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void scale(float *input, float *output, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * scale;
    }
}


__global__ void matrix_transpose(const float* input, float* output, int width, int height) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the index is within bounds
    if (row < height && col < width) {
        // Transpose the element
        output[col * height + row] = input[row * width + col];
    }
}

__global__ void naive_matrix_transpose(float* input, float* output, int numRows, int numCols) {
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < numRows * numCols) {
        int inputRow = globalThreadIdx / numCols;
        int inputCol = globalThreadIdx % numCols;
        int outputRow = inputCol;
        int outputCol = inputRow;
        output[outputRow * numRows + outputCol] = input[inputRow * numCols + inputCol];
    }
}

__global__ void print_2d_matrix_as_1d(const float* matrix, int width, int height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < width * height; i++) {
            //printf("%f ", matrix[i]);
        }
    }
}

int main()
{
    // print the model of GPU we are using
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU model: " << prop.name << std::endl;

    int32_t dimensions = DIM;
    int64_t nParticles = M;
    int32_t nCentroids = DIM;

    std::cout << "Particles: " << nParticles << " Centroids: " << nCentroids << " Dimensions: " << dimensions << std::endl;

    size_t size_p = nParticles * dimensions * sizeof(float);
    size_t size_c = nCentroids * dimensions * sizeof(float);
    double size_p_gb = (double) size_p / 1000000000;

    size_t size_out = nParticles * sizeof(int32_t);

    float *host_centroids = (float *)malloc(size_c);
    float *host_particles = (float *)malloc(size_p);

    srand(1337);
    for (int32_t i = 0; i < nCentroids * dimensions; i++)
    {
        host_centroids[i] = (float)rand() / RAND_MAX;
    }
    for (int32_t i = 0; i < nParticles * dimensions; i++)
    {
        host_particles[i] = (float)rand() / RAND_MAX;
    }

    float *centroids;
    cudaMalloc(&centroids, size_c);
    float *particles;
    cudaMalloc(&particles, size_p);
    int32_t *output;
    cudaMalloc(&output, size_out);

    float *centroids_gold_standard;
    cudaMalloc(&centroids_gold_standard, size_c);
    float *particles_gold_standard;
    cudaMalloc(&particles_gold_standard, size_p);
    int32_t *output_gold_standard;
    cudaMalloc(&output_gold_standard, size_out);
    
    cudaMemset(output, 0, size_out);
    cudaMemcpy(particles, host_particles, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids, host_centroids, size_c, cudaMemcpyHostToDevice);
    cudaMemset(output_gold_standard, 0, size_out);
    cudaMemcpy(particles_gold_standard, host_particles, size_p, cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_gold_standard, host_centroids, size_c, cudaMemcpyHostToDevice);
    
    // transpose particles before mmul
    // particles is nParticles x dimensions, row major and we need it in column major for cublas
    // so we need to transpose it

    float* transpose_particles;
    cudaMalloc(&transpose_particles, nParticles * dimensions * sizeof(float));

    float* transpose_centroids;
    cudaMalloc(&transpose_centroids, nCentroids * dimensions * sizeof(float));

    float* distances;
    cudaMalloc(&distances, nParticles * nCentroids * sizeof(float));
    cudaMemset(distances, 0, nParticles * nCentroids * sizeof(float));    

    // transpose distances afterwards
    float* transpose_distances;
    cudaMalloc(&transpose_distances, nParticles * nCentroids * sizeof(float));
   
    float *particles_squared;
    float* centroids_squared;
    cudaMalloc(&particles_squared, nParticles * sizeof(float));
    cudaMalloc(&centroids_squared, nCentroids * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float total_milliseconds = 0;

    // multiply transpose_particles * centroids
    // WMMA version
    half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    //transpose particles
    int transposeNumBlocks = (nParticles * dimensions + 255) / 256;
    int transposeNumThreads = 256;

    cudaEventRecord(start);
    naive_matrix_transpose<<<transposeNumBlocks, transposeNumThreads>>>(particles, transpose_particles, nParticles, dimensions);

    
    //transpose centroids
    int transposeNumBlocksCentroids = (nCentroids * dimensions + 255) / 256;
    int transposeNumThreadsCentroids = 256;

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nnaive_matrix_transpose 1 time: \t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);
    naive_matrix_transpose<<<transposeNumBlocksCentroids, transposeNumThreadsCentroids>>>(centroids, transpose_centroids, nCentroids, dimensions);

    // float2half conversion of transpose_particles and centroids
    int numThreadsConversion = 256;
    int numBlocksConversion = (nParticles * dimensions + numThreadsConversion - 1) / numThreadsConversion;

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nnaive_matrix_transpose 2 time: \t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    float2half<<<numBlocksConversion, numThreadsConversion>>>(particles, d_A, nParticles * dimensions);
    numBlocksConversion = (nCentroids * dimensions + numThreadsConversion - 1) / numThreadsConversion;

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nfloat2half 1 kernel time: \t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    float2half<<<numBlocksConversion, numThreadsConversion>>>(transpose_centroids, d_B, nCentroids * dimensions);

    // WMMA kernel configuration
    int wmmaThreadsPerBlock = 256;
    int wmmaBlocksPerGrid = (M / WMMA_M) * (N / WMMA_N);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nfloat2half 2 kernel time: \t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    wmma_matrix_multiply<<<wmmaBlocksPerGrid, wmmaThreadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nWMMA kernel time: \t\t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);


    int numThreadsScale = 256;
    int numBlocksScale = (M * N + numThreadsScale - 1) / numThreadsScale;
    // scale the result
    scale<<<numBlocksScale, numThreadsScale>>>(d_C, d_C, M * N, -2.0f);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nscale kernel time: \t\t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);


    matrix_transpose<<<dim3((nParticles + 15) / 16, (nCentroids + 15) / 16), dim3(16, 16)>>>(distances, transpose_distances, nParticles, nCentroids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nfinal transpose kernel time: \t\t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    l2_squared_norm_vec<<<nParticles, dimensions>>>(particles, particles_squared, dimensions, nParticles);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nl2_squared_norm_vec 1 kernel time: \t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    l2_squared_norm_vec<<<nParticles, dimensions>>>(centroids, centroids_squared, dimensions, nCentroids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nl2_squared_norm_vec 2 kernel time: \t\t\t" << milliseconds << " ms";
    cudaEventRecord(start);

    add_particles_l2_squared_to_distances<<<nParticles, nCentroids>>>(particles_squared, d_C, nParticles, nCentroids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nadd_particles_l2_squared_to_distances kernel time: \t" << milliseconds << " ms";
    cudaEventRecord(start);

    add_centroids_l2_squared_to_distances<<<nParticles, nCentroids>>>(centroids_squared, d_C, nParticles, nCentroids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nadd_centroids_l2_squared_to_distances kernel time: \t" << milliseconds << " ms";
    cudaEventRecord(start);

    find_argmin<<<nParticles, nCentroids>>>(d_C, output, nParticles, nCentroids);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    total_milliseconds += milliseconds;
    std::cout << "\nfind_argmin kernel time: \t" << milliseconds << " ms\n" << std::endl;
    cudaEventRecord(start);

    printf("Time for multi kernel version: %f ms\n", total_milliseconds);
    printf("Input data / execution time: %f GB/s\n", size_p_gb / (total_milliseconds / 1000));


    // copy output back to host and check
    int32_t *host_output_matmul = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output_matmul, output, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts_matmul = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++)
    {
        int32_t particle = host_output_matmul[i];
        if (particle < 0 || particle >= nCentroids)
        {
            std::cout << "bad" << std::endl;
            return 1;
        }
        counts_matmul[particle]++;
    }





    int numThreads = 256;
    int numBlocks = (nParticles + numThreads - 1) / numThreads;

    assign_labels_very_slowly<<<numBlocks, numThreads>>>(centroids_gold_standard, particles_gold_standard, output_gold_standard, dimensions, nParticles, nCentroids);

    int32_t *host_output = (int32_t *)malloc(size_out);
    cudaMemcpy(host_output, output_gold_standard, size_out, cudaMemcpyDeviceToHost);

    int32_t *counts = (int32_t *)calloc(nCentroids, sizeof(int32_t));
    for (int32_t i = 0; i < nParticles; i++)
    {
        int32_t particle = host_output[i];
        if (particle < 0 || particle >= nCentroids)
        {
            std::cout << "bad" << std::endl;
            return 1;
        }
        counts[particle]++;
    }
    for (int32_t i = 0; i < nCentroids; i++)
    {
        //std::cout << counts[i] << " vs " << counts_matmul[i] << " to centroid " << i << std::endl;
    }
}