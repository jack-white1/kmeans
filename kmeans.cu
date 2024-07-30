#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;

#define M 100000000
#define N 16
#define K 16

// Define the dimensions for WMMA
const long int WMMA_M = 16;
const long int WMMA_N = 16;
const long int WMMA_K = 16;

#define NBLOCKS 2048
#define NTHREADS 32

//#define DEBUG 1

__global__ void assign_labels(float* centroids, float* particles, int32_t* output, int32_t dim, int32_t nParticles, int32_t nCentroids) {
    __shared__ half        centroidsTile[K*N];
    __shared__ half        particlesTile[K*N];
    __shared__ half           outputTile[K*N];

    __shared__ half        centroidsSquaredTileHalf[K*N];
	__shared__ half				 sumSquaredTileHalf[K*N];

    long int rowsPerBlock = M / NBLOCKS;

    for (long int loadStep = 0; loadStep < K * N / NTHREADS; loadStep++){
        long int localIndex = loadStep * NTHREADS + threadIdx.x;
        centroidsTile[localIndex] = __float2half(centroids[localIndex]);
    }

    if (threadIdx.x < 16){
        half mySquaredVal = __float2half(0.0f);
        half newValue;
		
        for (long int step = 0; step < 16; step++){
            newValue = centroidsTile[threadIdx.x * 16 + step];
            mySquaredVal += newValue * newValue;
        }

		
        for (long int step = 0; step < 16; step++){
			centroidsSquaredTileHalf[threadIdx.x * 16 + step]=mySquaredVal;
		} 
    }
    
    for (long int tileIndex = 0; tileIndex < M / NBLOCKS / K; tileIndex++){
        long int particlesOffset = blockIdx.x * rowsPerBlock * 16 + tileIndex*16*16;

		float4* startingPointer4 = reinterpret_cast<float4*>(&(particles[particlesOffset]));
		float4 temp1, temp2;

		temp1 = startingPointer4[threadIdx.x];
		temp2 = startingPointer4[threadIdx.x + 32];

		particlesTile[threadIdx.x * 4] = __float2half(temp1.x);
		particlesTile[threadIdx.x * 4 + 1] = __float2half(temp1.y);
		particlesTile[threadIdx.x * 4 + 2] = __float2half(temp1.z);
		particlesTile[threadIdx.x * 4 + 3] = __float2half(temp1.w);
		particlesTile[threadIdx.x * 4 + 128] = __float2half(temp2.x);
		particlesTile[threadIdx.x * 4 + 129] = __float2half(temp2.y);
		particlesTile[threadIdx.x * 4 + 130] = __float2half(temp2.z);
		particlesTile[threadIdx.x * 4 + 131] = __float2half(temp2.w);

        if (threadIdx.x < 16){
            half mySquaredVal = __float2half(0.0f);
            half newValue;
			
            for (long int step = 0; step < 16; step++){
                newValue = particlesTile[threadIdx.x * 16 + step];
                mySquaredVal += newValue * newValue;
            }
			
            for (long int step = 0; step < 16; step++){
				sumSquaredTileHalf[step * 16 + threadIdx.x] = __float2half(-0.5) * (mySquaredVal + centroidsSquaredTileHalf[step * 16 + threadIdx.x]);
			}
        }

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
        wmma::load_matrix_sync(a_frag, particlesTile, 16);
        wmma::load_matrix_sync(b_frag, centroidsTile, 16);
        wmma::load_matrix_sync(c_frag, sumSquaredTileHalf, 16, wmma::mem_col_major);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(outputTile, c_frag, 16, wmma::mem_row_major);

		if (threadIdx.x < 16){
			long int minIndex = 0;
			half minDistance = __habs(outputTile[threadIdx.x * 16]);
			half tempDistance;
			
			for (long int step = 1; step < 16; step++){
				tempDistance = __habs(outputTile[threadIdx.x * 16 + step]);
				if (tempDistance < minDistance){
					minDistance = tempDistance;
					minIndex = step;
				}
			}
			output[blockIdx.x * rowsPerBlock + tileIndex * 16 + threadIdx.x] = minIndex;
		}
    }
}

int main() {
	// print GPU model name
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "GPU model: " << prop.name << std::endl;

	int32_t dim = 16;
	int64_t nParticles = M;
	int32_t nCentroids = 16;
	std::cout << "dim " << dim << " nParticles " << nParticles << " nCentroids " << nCentroids << std::endl;

	size_t size_p = nParticles * dim * sizeof(float);
	size_t size_c = nCentroids * dim * sizeof(float);

	size_t size_out = nParticles * sizeof(int32_t);

	float* host_centroids = (float*) malloc(size_c);
	float* host_particles = (float*) malloc(size_p);

	srand(1337);
	for (int32_t i = 0; i < nCentroids * dim; i++) {
		host_centroids[i] = (float) rand() / RAND_MAX;
	}
	for (int32_t i = 0; i < nParticles * dim; i++) {
		host_particles[i] = (float) rand() / RAND_MAX;
	}

	cudaError_t err = cudaSuccess;

	float* centroids;
	err = cudaMalloc(&centroids, size_c);
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	float* particles;
	err=cudaMalloc(&particles, size_p);
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	int32_t* output;
	err=cudaMalloc(&output, size_out);
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaMemset(output, 0, size_out);
	cudaMemcpy(particles, host_particles, size_p, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaMemcpy(centroids, host_centroids, size_c, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaDeviceSynchronize();

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	auto startCPU = std::chrono::high_resolution_clock::now();
	cudaEventRecord(start);

	long int numRuns = 100;
	for (long int run = 0; run < numRuns; run++) assign_labels<<<NBLOCKS, NTHREADS>>>(centroids, particles, output, dim, nParticles, nCentroids);
	
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaDeviceSynchronize();
	auto endCPU = std::chrono::high_resolution_clock::now();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
	std::cout << (milliseconds*1000)/numRuns << " average us gpu" << std::endl;
	std::cout << duration.count()/numRuns << " average us cpu" << std::endl;
	double memLoaded = size_p; // approx
	std::cout << "memloaded " << memLoaded << std::endl;
	std::cout << "bw gb/sec " << numRuns * memLoaded / 1000 / duration.count() << std::endl; // gigabyte = billion but microsecond = million so it comes out to thousand
	std::cout << "flops " << (double) numRuns * (double) nParticles * (double) nCentroids * (double) dim / ((double) duration.count() / 1000000) << std::endl; // approx number of floating polong int multiplies per second
	cudaEventDestroy(start);
	cudaEventDestroy(end);


	int32_t* host_output = (int32_t*) malloc(size_out);
	cudaMemcpy(host_output, output, size_out, cudaMemcpyDeviceToHost);

	int32_t* counts = (int32_t*) calloc(nCentroids, sizeof(int32_t));
	for (int32_t i = 0; i < nParticles; i++) {
		int32_t particle = host_output[i];
		if (particle < 0 || particle >= nCentroids) {
			std::cout << "bad" << std::endl;
			return 1;
		}
		counts[particle]++;
	}
	for (int32_t i = 0; i < nCentroids; i++) {
		std::cout << counts[i] << " particles were assigned to centroid " << i << std::endl;
	}
    return 0;
}