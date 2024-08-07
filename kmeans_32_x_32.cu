#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;

#define M 8388608
#define N 32
#define K 32

// Define the dimensions for WMMA
const long int WMMA_M = 16;
const long int WMMA_N = 16;
const long int WMMA_K = 16;

#define NBLOCKS 2048
#define NTHREADS 128

//#define DEBUG 1
//#define TIMING 1
#define DOUBLE_BUFFER 1
//#define MINIBATCH 1

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
            for (int32_t dimension = 0; dimension < dimensions; dimension++)
            {
                float d = centroids[j * dimensions + dimension] - particles[idx * dimensions + dimension];
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


__global__ void assign_labels(float* centroids, float* particles, float* newCentroids, int32_t* globalCentroidCounts, int32_t* output, int32_t dim, int32_t nParticles, int32_t nCentroids) {
#ifdef TIMING
	unsigned long int startKernel, stopKernel, totalKernel;
	unsigned long int startLoad, stopLoad, totalLoad;
	unsigned long int startSumSquared, stopSumSquared, totalSumSquared;
	unsigned long int startSumSquared2, stopSumSquared2, totalSumSquared2;
	unsigned long int startMMA, stopMMA, totalMMA;
	unsigned long int startStore, stopStore, totalStore;
	startKernel = clock64();
	totalKernel = 0;
	totalLoad = 0;
	totalSumSquared = 0;
	totalSumSquared2 = 0;
	totalMMA = 0;
	totalStore = 0;
#endif
	__shared__ half        centroidsTile[K][N];
    __shared__ half        particlesTile[K][N];
    __shared__ half           outputTile[K][N];
	__shared__ half 	   mmaOutputTile[K][N];
#ifdef MINIBATCH
	__shared__ half 	newCentroidsTile[K][N];
#endif

#ifdef DOUBLE_BUFFER
	__shared__ float   nextParticlesTile[K][N];
#endif

	__shared__ int32_t       centroidCounts[N];

	if (threadIdx.x < N){
		centroidCounts[threadIdx.x] = 0;
	}

    long int rowsPerBlock = M / NBLOCKS;
	long int tilesPerBlock = M / NBLOCKS / K;

	int warpIndex = threadIdx.x / 32;

#ifdef DEBUG
	int targetThreadIdx = 0;
	int targetBlockIdx = 1;
	int targetTileIndex = 10;
	if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx){
		printf("rowsPerBlock: %ld, tilesPerBlock: %ld\n", rowsPerBlock, tilesPerBlock);
	}
#endif

	// transpose the centroids tile on the load
	if (warpIndex == 0){
		for (int i = 0; i < K; i++){
			centroidsTile[threadIdx.x][i] = __float2half(centroids[i * N + threadIdx.x]);
		}
	}

#ifdef MINIBATCH
	// set newCentroids to zero
	for (int i = 0; i < K; i++){
		newCentroidsTile[i][threadIdx.x] = __float2half(0.0f);
	}
#endif

    half myCentroidSquaredVal = __float2half(0.0f);
    half newCentroidValue;

	__syncthreads();
	for (long int step = 0; step < 32; step++){
		newCentroidValue = centroidsTile[step][threadIdx.x%32];
		myCentroidSquaredVal += newCentroidValue * newCentroidValue;
	}


#ifdef DEBUG
	if (blockIdx.x == targetBlockIdx){
		printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
		printf("threadIdx.x: %d, myCentroidSquaredVal: %f\n", threadIdx.x,__half2float(myCentroidSquaredVal));
	}
#endif

    for (long int tileIndex = 0; tileIndex < tilesPerBlock; tileIndex++){
        long int particlesOffset = blockIdx.x * rowsPerBlock * K + tileIndex*K*K;

#ifdef DEBUG
		if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Particles Offset: %ld\n", particlesOffset);
		}
#endif

#ifdef TIMING
		startLoad = clock64();
#endif

#ifndef DOUBLE_BUFFER
		// Directly load the tile
		float4* startingPointer4 = reinterpret_cast<float4*>(&(particles[particlesOffset]));
		float4 temp0 = startingPointer4[threadIdx.x];
		float4 temp1 = startingPointer4[threadIdx.x + 128];

		int index0, index1, index2, index3, index4, index5, index6, index7;
		index0 = 4 * threadIdx.x;
		index1 = 4 * threadIdx.x + 1;
		index2 = 4 * threadIdx.x + 2;
		index3 = 4 * threadIdx.x + 3;
		index4 = 4 * threadIdx.x + 512;
		index5 = 4 * threadIdx.x + 513;
		index6 = 4 * threadIdx.x + 514;
		index7 = 4 * threadIdx.x + 515;
		
		particlesTile[index0 / 32][index0 % 32] = __float2half(temp0.x);
		particlesTile[index1 / 32][index1 % 32] = __float2half(temp0.y);
		particlesTile[index2 / 32][index2 % 32] = __float2half(temp0.z);
		particlesTile[index3 / 32][index3 % 32] = __float2half(temp0.w);
		particlesTile[index4 / 32][index4 % 32] = __float2half(temp1.x);
		particlesTile[index5 / 32][index5 % 32] = __float2half(temp1.y);
		particlesTile[index6 / 32][index6 % 32] = __float2half(temp1.z);
		particlesTile[index7 / 32][index7 % 32] = __float2half(temp1.w);
#endif


#ifdef DOUBLE_BUFFER
		int row = threadIdx.x / 32;
		int col = threadIdx.x % 32;

		int row2 = threadIdx.x / 32 + 4;
		int col2 = col;
		if (tileIndex == 0) {
            // Directly load the first tile
            float4* startingPointer4 = reinterpret_cast<float4*>(&(particles[particlesOffset]));
            float4 temp0 = startingPointer4[threadIdx.x];
            float4 temp1 = startingPointer4[threadIdx.x + 128];

            particlesTile[row][col] = __float2half(temp0.x);
            particlesTile[row][col + 1] = __float2half(temp0.y);
			particlesTile[row][col + 2] = __float2half(temp0.z);
			particlesTile[row][col + 3] = __float2half(temp0.w);
			particlesTile[row2][col2] = __float2half(temp1.x);
			particlesTile[row2][col2 + 1] = __float2half(temp1.y);
			particlesTile[row2][col2 + 2] = __float2half(temp1.z);
			particlesTile[row2][col2 + 3] = __float2half(temp1.w);

            // Start asynchronous copy for the next tile
            __pipeline_memcpy_async(&(nextParticlesTile[row][col]), &(particles[particlesOffset + 32 * 32 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 4][col]), &(particles[particlesOffset + 32 * 32 + 128 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 8][col]), &(particles[particlesOffset + 32 * 32 + 256 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 12][col]), &(particles[particlesOffset + 32 * 32 + 384 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 16][col]), &(particles[particlesOffset + 32 * 32 + 512 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 20][col]), &(particles[particlesOffset + 32 * 32 + 640 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 24][col]), &(particles[particlesOffset + 32 * 32 + 768 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 28][col]), &(particles[particlesOffset + 32 * 32 + 896 + threadIdx.x]), 4, 0);
            __pipeline_commit();

        } else {
            // Wait for the previous copy to complete
            __pipeline_wait_prior(0);

            // Use the data from nextParticlesTile
			particlesTile[row][col] =    __float2half(nextParticlesTile[row][col]);
			particlesTile[row+4][col] =  __float2half(nextParticlesTile[row + 4][col]);
			particlesTile[row+8][col] =  __float2half(nextParticlesTile[row + 8][col]);
			particlesTile[row+12][col] = __float2half(nextParticlesTile[row + 12][col]);
			particlesTile[row+16][col] = __float2half(nextParticlesTile[row + 16][col]);
			particlesTile[row+20][col] = __float2half(nextParticlesTile[row + 20][col]);
			particlesTile[row+24][col] = __float2half(nextParticlesTile[row + 24][col]);
			particlesTile[row+28][col] = __float2half(nextParticlesTile[row + 28][col]);

            // Start asynchronous copy for the next tile
			__pipeline_memcpy_async(&(nextParticlesTile[row][col]),      &(particles[particlesOffset + 32 * 32 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 4][col]),  &(particles[particlesOffset + 32 * 32 + 128 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 8][col]),  &(particles[particlesOffset + 32 * 32 + 256 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 12][col]), &(particles[particlesOffset + 32 * 32 + 384 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 16][col]), &(particles[particlesOffset + 32 * 32 + 512 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 20][col]), &(particles[particlesOffset + 32 * 32 + 640 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 24][col]), &(particles[particlesOffset + 32 * 32 + 768 + threadIdx.x]), 4, 0);
			__pipeline_memcpy_async(&(nextParticlesTile[row + 28][col]), &(particles[particlesOffset + 32 * 32 + 896 + threadIdx.x]), 4, 0);
            __pipeline_commit();
        }
#endif

#ifdef TIMING
		stopLoad = clock64();
		totalLoad += stopLoad - startLoad;
		startSumSquared = clock64();
#endif

		half mySquaredVal = __float2half(0.0f);
		half newValue, outValue;

		if (warpIndex == 0){
			for (long int step = 0; step < K; step++){
				newValue = particlesTile[threadIdx.x][step];
				mySquaredVal += newValue * newValue;
			}

#ifdef DEBUG
			if (blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
				printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
				printf("threadIdx.x: %d, myParticlesSquaredVal: %f\n", threadIdx.x,__half2float(mySquaredVal));
			}
#endif

			for (long int step = 0; step < K; step++){
				outValue = __float2half(-0.5) * (mySquaredVal + __shfl_sync(0xffffffff, myCentroidSquaredVal, step));
				outputTile[threadIdx.x][step] = outValue;
			}
		}

#ifdef TIMING
		stopSumSquared = clock64();
		totalSumSquared += stopSumSquared - startSumSquared;
		startSumSquared2 = clock64();
#endif

		/*
		__syncthreads();
		// non warpzero version
		#pragma unroll
		for (int step = 0; step < 8; step++){
			int myRow = (threadIdx.x / 32) + 4 * step;
			int myCol = threadIdx.x % 32;
			half myVal = particlesTile[myRow][myCol];
			half myValSquared = myVal * myVal;
			// warp sum reduce myValSquared
			for (int i = 16; i > 0; i /= 2){
				myValSquared += __shfl_down_sync(0xffffffff, myValSquared, i);
			}
			myValSquared = __shfl_sync(0xffffffff, myValSquared, 0);
			// calculate the output value
			half outVal = __float2half(-0.5) * (myValSquared + myCentroidSquaredVal);
			outputTile[myRow][myCol] = outVal;
		}
		__syncthreads();
		*/

#ifdef TIMING
		stopSumSquared2 = clock64();
		totalSumSquared2 += stopSumSquared2 - startSumSquared2;
		startMMA = clock64();
#endif
#ifdef DEBUG
		if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
			printf("\nCentroids Tile 32x32: (on wmma entry)\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(centroidsTile[i][j]));
				}
				printf("\n");
			}
			printf("\nParticles Tile 32x32: (on wmma entry)\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(particlesTile[i][j]));
				}
				printf("\n");
			}
			printf("\nOutput Tile 32x32: (on wmma entry)\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(outputTile[i][j]));
				}
				printf("\n");
			}
		}
#endif

		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

		// warp 0 to do the first two
		__syncthreads();
		if (warpIndex == 0){
			// top left corner of A, top left corner of B, C = 0.0f, top left corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[0][0]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[0][0]), 32);
			wmma::load_matrix_sync(c_frag, &(outputTile[0][0]), 32, wmma::mem_row_major);
			mma_sync(c_frag, a_frag, b_frag, c_frag);

			// top right corner of A, bottom left corner of B, top left corner of C, top left corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[0][16]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[16][0]), 32);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			wmma::store_matrix_sync(&(mmaOutputTile[0][0]), c_frag, 32, wmma::mem_row_major);
		}

		// warp 1 to do the next two
		if (warpIndex == 1){
			// top left corner of A, top right corner of B, C = 0.0f, top right corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[0][0]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[0][16]), 32);
			wmma::load_matrix_sync(c_frag, &(outputTile[0][16]), 32, wmma::mem_row_major);
			mma_sync(c_frag, a_frag, b_frag, c_frag);

			// top right corner of A, bottom right corner of B, top right corner of C, top right corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[0][16]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[16][16]), 32);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			wmma::store_matrix_sync(&(mmaOutputTile[0][16]), c_frag, 32, wmma::mem_row_major);
		}

		// warp 2 to do the next two
		if (warpIndex == 2){
			// bottom left corner of A, top left corner of B, C = 0.0f, bottom left corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[16][0]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[0][0]), 32);
			wmma::load_matrix_sync(c_frag, &(outputTile[16][0]), 32, wmma::mem_row_major);
			mma_sync(c_frag, a_frag, b_frag, c_frag);

			// bottom right corner of A, bottom left corner of B, bottom left corner of C, bottom left corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[16][16]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[16][0]), 32);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			wmma::store_matrix_sync(&(mmaOutputTile[16][0]), c_frag, 32, wmma::mem_row_major);
		}

		// warp 3 to do the last two
		if (warpIndex == 3){
			// bottom left corner of A, top right corner of B, C = 0.0f, bottom right corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[16][0]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[0][16]), 32);
			wmma::load_matrix_sync(c_frag, &(outputTile[16][16]), 32, wmma::mem_row_major);
			mma_sync(c_frag, a_frag, b_frag, c_frag);

			// bottom right corner of A, bottom right corner of B, bottom right corner of C, bottom right corner of output
			wmma::load_matrix_sync(a_frag, &(particlesTile[16][16]), 32);
			wmma::load_matrix_sync(b_frag, &(centroidsTile[16][16]), 32);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			wmma::store_matrix_sync(&(mmaOutputTile[16][16]), c_frag, 32, wmma::mem_row_major);
		}
		__syncthreads();

#ifdef TIMING
		stopMMA = clock64();
		totalMMA += stopMMA - startMMA;
		startStore = clock64();
#endif

		if (warpIndex == 0){
#ifdef DEBUG
		if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Output Tile 32x32:\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(mmaOutputTile[i][j]));
				}
				printf("\n");
			}
		}
#endif

			long int minIndex = 0;
			half minDistance = __habs(mmaOutputTile[threadIdx.x][0]);
			half tempDistance;

			for (long int step = 1; step < 32; step++){
				tempDistance = __habs(mmaOutputTile[threadIdx.x][step]);
				if (tempDistance < minDistance){
					minDistance = tempDistance;
					minIndex = step;
				}
			}
			output[blockIdx.x * rowsPerBlock + tileIndex * 32 + threadIdx.x] = minIndex;
#ifdef MINIBATCH
			int centroidOutIndex;
			for (int i = 0; i < 32; i++){
				centroidOutIndex = __shfl_sync(0xffffffff, minIndex, i);
				newCentroidsTile[centroidOutIndex][threadIdx.x] += particlesTile[centroidOutIndex][threadIdx.x];
				if (threadIdx.x == 0) centroidCounts[centroidOutIndex] += 1;
			}
#endif
		}

#ifdef TIMING
		stopStore = clock64();
		totalStore += stopStore - startStore;
#endif


#ifdef DEBUG
			if (blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
				printf("%d , from thread %d to output %ld\n", output[blockIdx.x * rowsPerBlock + tileIndex * 32 + threadIdx.x], threadIdx.x, blockIdx.x * rowsPerBlock + tileIndex * 32 + threadIdx.x);
			}
#endif
    }
#ifdef TIMING
	stopKernel = clock64();
	totalKernel += stopKernel - startKernel;
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("Kernel Time: %lu\n", totalKernel);
		printf("Load Time: %lu\n", totalLoad);
		printf("Sum Squared Time: %lu\n", totalSumSquared);
		printf("Sum Squared 2 Time: %lu\n", totalSumSquared2);
		printf("MMA Time: %lu\n", totalMMA);
		printf("Store Time: %lu\n", totalStore);
		printf("\n");
	}
#endif

#ifdef MINIBATCH
	// write out new centroids
	if (warpIndex == 0){
		for (int i = 0; i < K; i++){
			newCentroids[blockIdx.x * K * N + i * K + threadIdx.x] = __half2float(newCentroidsTile[i][threadIdx.x]);
		}
		globalCentroidCounts[blockIdx.x * N + threadIdx.x] = centroidCounts[threadIdx.x];
	}
#endif
}

#ifdef MINIBATCH
// assuming blockdim = (32,32)
// assuming griddim = (1,1)
__global__ void reduceCentroids(float* centroidsExtendedArray, float* newCentroids, int32_t* centroidCountsExtendedArray, int32_t* centroidCountsReduced){
	float newCentroidsTile[K][N];
	int32_t centroidCountsVector[N];
	
	if (threadIdx.y == 0) centroidCountsVector[threadIdx.x] = 0;
	__syncthreads();

	newCentroidsTile[threadIdx.y][threadIdx.x] = 0.0f;

	for (int i = 0 ; i < 2048; i++){
		newCentroidsTile[threadIdx.y][threadIdx.x] += newCentroids[i * K * N + threadIdx.y * K + threadIdx.x];
		if (threadIdx.y == 0){
			centroidCountsVector[threadIdx.x] += centroidCountsExtendedArray[i * N + threadIdx.x];
		}
	}
	__syncthreads();

	newCentroidsTile[threadIdx.y][threadIdx.x] /= (float)centroidCountsVector[threadIdx.y];
	newCentroids[threadIdx.y*K + threadIdx.x] = newCentroidsTile[threadIdx.y][threadIdx.x];

	__syncthreads();
	if (threadIdx.y == 0 ) centroidCountsReduced[threadIdx.x] = centroidCountsVector[threadIdx.x];

}
#endif

int main() {
	// print GPU model name
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "GPU model: " << prop.name << std::endl;

	int32_t dim = K;
	int64_t nParticles = M;
	int32_t nCentroids = N;

	std::cout << "dim " << dim << " nParticles " << nParticles << " nCentroids " << nCentroids << std::endl;

	size_t size_p = nParticles * dim * sizeof(float);
	size_t size_c = nCentroids * dim * sizeof(float);

	size_t size_out = nParticles * sizeof(int32_t);

	float* host_centroids = (float*) malloc(size_c);
	float* host_particles = (float*) malloc(size_p);

	srand(1337);
#ifdef DEBUG
	printf("Centroids:\n");
#endif
	for (int32_t i = 0; i < nCentroids * dim; i++) {
		host_centroids[i] = (float) rand() / RAND_MAX;
#ifdef DEBUG
		printf("%f ", host_centroids[i]);
		if (i % dim == dim - 1) {
			printf("\n");
		}
#endif
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


	float* newCentroids;
	err = cudaMalloc(&newCentroids, size_c * NBLOCKS);
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}


	int32_t* centroidCounts;
	err = cudaMalloc(&centroidCounts, nCentroids * sizeof(int32_t) * NBLOCKS);
	err = cudaMemset(centroidCounts, 0, nCentroids * sizeof(int32_t) * NBLOCKS);
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

	int32_t* output_slow;
	err=cudaMalloc(&output_slow, size_out);
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaMemset(output, 0, size_out);
	cudaMemset(output_slow, 0, size_out);
	cudaMemcpy(particles, host_particles, size_p, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaMemcpy(centroids, host_centroids, size_c, cudaMemcpyHostToDevice);
	std::cout << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	cudaDeviceSynchronize();


	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	long int numRuns = 100;
	float totalSeconds = 0.0f;
	float milliseconds = 0.0f;
#ifdef DEBUG
	numRuns = 1;
#endif

	for (long int run = 0; run < numRuns; run++) {
		cudaEventRecord(start);
		assign_labels<<<NBLOCKS, NTHREADS, 32000>>>(centroids, particles, newCentroids, centroidCounts, output, dim, nParticles, nCentroids);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&milliseconds, start, end);
		totalSeconds += milliseconds/1000;
		milliseconds = 0.0f;
	}

	int32_t* centroidCountsReduced;
	err = cudaMalloc(&centroidCountsReduced, nCentroids * sizeof(int32_t));
	err = cudaMemset(centroidCountsReduced, 0, nCentroids * sizeof(int32_t));
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}

#ifdef MINIBATCH
	cudaDeviceSynchronize();
	reduceCentroids<<<1, dim3(32,32)>>>(centroids, newCentroids, centroidCounts, centroidCountsReduced);
	cudaDeviceSynchronize();
#endif

	std::cout << 1000*(totalSeconds / numRuns) << " ms average kernel execution time" << std::endl;
	double memLoaded = numRuns * size_p; // approx
	std::cout << "Bandwidth: " << memLoaded / totalSeconds / 1000 / 1000 / 1000 <<" GB/s" << std::endl; // gigabyte = billion but microsecond = million so it comes out to thousand
	std::cout << "Compute: " << (double) numRuns * 2.0 * (double) nParticles * (double) nCentroids * (double) dim / (double) totalSeconds / 1e12 << " TFLOPS" << std::endl; // approx number of floating polong int multiplies per second
	

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	assign_labels_very_slowly<<<(nParticles + 1 - NTHREADS) / NTHREADS, NTHREADS>>>(centroids, particles, output_slow, dim, nParticles, nCentroids);
	cudaDeviceSynchronize();

	int32_t* host_output = (int32_t*) malloc(size_out);
	cudaMemcpy(host_output, output, size_out, cudaMemcpyDeviceToHost);

	int32_t* host_output_slow = (int32_t*) malloc(size_out);
	cudaMemcpy(host_output_slow, output_slow, size_out, cudaMemcpyDeviceToHost);

	int32_t* counts = (int32_t*) calloc(nCentroids, sizeof(int32_t));
	int32_t* counts_slow = (int32_t*) calloc(nCentroids, sizeof(int32_t));
	for (int32_t i = 0; i < nParticles; i++) {
		int32_t particle = host_output[i];
		if (particle < 0 || particle >= nCentroids) {
			std::cout << "bad" << std::endl;
			return 1;
		}
		counts[particle]++;
		//if (particle != i % 32) printf("Particle %d assigned to centroid %d\n", i, particle);
		int32_t particle_slow = host_output_slow[i];
		if (particle_slow < 0 || particle_slow >= nCentroids) {
			std::cout << "bad" << std::endl;
			return 1;
		}
		counts_slow[particle_slow]++;
	}
	for (int32_t i = 0; i < nCentroids; i++) {
		std::cout << counts[i] << " (fast) vs " << counts_slow[i] << " (slow) particles were assigned to centroid " << i << std::endl;
	}

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return 1;
	}
    return 0;
}