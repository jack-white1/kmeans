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
#define TIMING 1

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
		//if (idx < 32){
		//	printf("OLD METHOD Particle %d assigned to centroid %d\n", idx, closestCentroidIdx);
		//}
    }
}

__global__ void assign_labels(float* centroids, float* particles, int32_t* output, int32_t dim, int32_t nParticles, int32_t nCentroids) {
	__shared__ half        centroidsTile[K*N];
    __shared__ half        particlesTile[K*N];
    __shared__ half           outputTile[K*N];

    long int rowsPerBlock = M / NBLOCKS;
	long int tilesPerBlock = M / NBLOCKS / K;

#ifdef TIMING
	unsigned long int startKernel, stopKernel, totalKernel;
	unsigned long int startLoad, stopLoad, totalLoad;
	unsigned long int startStore, stopStore, totalStore;
	unsigned long int startMatmul, stopMatmul, totalMatmul;
	unsigned long int startSumsq, stopSumsq, totalSumsq;
	unsigned long int startMmaLoad, stopMmaLoad, totalMmaLoad;
	unsigned long int startMmaSync, stopMmaSync, totalMmaSync;
	unsigned long int startMmaStore, stopMmaStore, totalMmaStore;
	totalKernel = 0;
	totalLoad = 0;
	totalStore = 0;
	totalMatmul = 0;
	totalSumsq = 0;
	totalMmaLoad = 0;
	totalMmaSync = 0;
	totalMmaStore = 0;
	startKernel = clock64();
#endif
	
#ifdef DEBUG
	int targetThreadIdx = 0;
	int targetBlockIdx = 1;
	int targetTileIndex = 10;
	if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx){
		printf("rowsPerBlock: %ld, tilesPerBlock: %ld\n", rowsPerBlock, tilesPerBlock);
	}
#endif

	// transpose the tile on the load
	if (threadIdx.x < 32){
		for (int i = 0; i < K; i++){
			centroidsTile[threadIdx.x * N + i] = __float2half(centroids[i * N + threadIdx.x]);
		}
	}

    half myCentroidSquaredVal = __float2half(0.0f);
    half newCentroidValue;

	if (threadIdx.x < 32){
		for (long int step = 0; step < 32; step++){
			newCentroidValue = centroidsTile[step * 32 + threadIdx.x];
			myCentroidSquaredVal += newCentroidValue * newCentroidValue;
		}
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

		float4* startingPointer4 = reinterpret_cast<float4*>(&(particles[particlesOffset]));
		float4 temp[2];

#ifdef TIMING
		startLoad = clock64();
#endif
		for (int i = 0; i < 2; i++){
			temp[i] = startingPointer4[i*128 + threadIdx.x];
			particlesTile[threadIdx.x * 4 + 512 * i] = __float2half(temp[i].x);
			particlesTile[threadIdx.x * 4 + 512 * i + 1] = __float2half(temp[i].y);
			particlesTile[threadIdx.x * 4 + 512 * i + 2] = __float2half(temp[i].z);
			particlesTile[threadIdx.x * 4 + 512 * i + 3] = __float2half(temp[i].w);
		}

#ifdef TIMING
		stopLoad = clock64();
		totalLoad += stopLoad - startLoad;
#endif

		half mySquaredVal = __float2half(0.0f);
		half newValue, outValue;
		
		if (threadIdx.x < 32){
#ifdef TIMING
			startSumsq = clock64();
#endif
			for (long int step = 0; step < K; step++){
				newValue = particlesTile[threadIdx.x * K + step];
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
				outputTile[threadIdx.x * K + step] = outValue;
			}
		}	
#ifdef TIMING
		stopSumsq = clock64();
		totalSumsq += stopSumsq - startSumsq;
#endif

#ifdef DEBUG
		if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Centroids Tile 32x32 (should be transposed vs GMEM version):\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(centroidsTile[i * 32 + j]));
				}
				printf("\n");
			}
			printf("\n");
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Particles Tile 32x32:\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(particlesTile[i * 32 + j]));
				}
				printf("\n");
			}
			printf("\n");
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Sum Squared Tile 32x32:\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(outputTile[i * 32 + j]));
				}
			printf("\n");
			}
		}
#endif

		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

#ifdef TIMING
		startMatmul = clock64();
#endif
		int warpIndex = threadIdx.x / 32;
		// warp 0 to do the first two
		__syncthreads();
		if (warpIndex == 0){
			// top left corner of A, top left corner of B, C = 0.0f, top left corner of output
#ifdef TIMING
			startMmaLoad = clock64();
#endif
			wmma::load_matrix_sync(a_frag, particlesTile, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile, 32);
			wmma::load_matrix_sync(c_frag, outputTile, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
			startMmaLoad = clock64();
#endif
			
			// top right corner of A, bottom left corner of B, top left corner of C, top left corner of output
			wmma::load_matrix_sync(a_frag, particlesTile + 16, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 32*16, 32);
			wmma::load_matrix_sync(c_frag, outputTile, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;	
#endif
		}

		// warp 1 to do the next two
		if (warpIndex == 0){
			// top left corner of A, top right corner of B, C = 0.0f, top right corner of output
#ifdef TIMING
			startMmaLoad = clock64();
#endif
			wmma::load_matrix_sync(a_frag, particlesTile, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 16, 32);
			wmma::load_matrix_sync(c_frag, outputTile+16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile+16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
			startMmaLoad = clock64();
#endif

			// top right corner of A, bottom right corner of B, top right corner of C, top right corner of output
			wmma::load_matrix_sync(a_frag, particlesTile + 16, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 32*16 + 16, 32);
			wmma::load_matrix_sync(c_frag, outputTile+16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile+16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
#endif
		}

		// warp 2 to do the next two
		if (warpIndex == 0){
			// bottom left corner of A, top left corner of B, C = 0.0f, bottom left corner of output
#ifdef TIMING
			startMmaLoad = clock64();
#endif
			wmma::load_matrix_sync(a_frag, particlesTile + 16*32, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile, 32);
			wmma::load_matrix_sync(c_frag, outputTile + 32*16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile + 32 * 16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
			startMmaLoad = clock64();
#endif
			// bottom right corner of A, bottom left corner of B, bottom left corner of C, bottom left corner of output
			wmma::load_matrix_sync(a_frag, particlesTile + 16*32 + 16, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 32*16, 32);
			wmma::load_matrix_sync(c_frag, outputTile + 32 * 16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile + 32 * 16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
#endif
		}

		// warp 3 to do the last two
		if (warpIndex == 0){
			// bottom left corner of A, top right corner of B, C = 0.0f, bottom right corner of output
#ifdef TIMING
			startMmaLoad = clock64();
#endif
			wmma::load_matrix_sync(a_frag, particlesTile + 16*32, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 16, 32);
			wmma::load_matrix_sync(c_frag, outputTile + 32 * 16 + 16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile + 32 * 16 + 16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
			startMmaLoad = clock64();
#endif

			// bottom right corner of A, bottom right corner of B, bottom right corner of C, bottom right corner of output
			wmma::load_matrix_sync(a_frag, particlesTile + 16*32 + 16, 32);
			wmma::load_matrix_sync(b_frag, centroidsTile + 16*32 + 16, 32);
			wmma::load_matrix_sync(c_frag, outputTile + 32 * 16 + 16, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaLoad = clock64();
			totalMmaLoad += stopMmaLoad - startMmaLoad;
			startMmaSync = clock64();
#endif
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#ifdef TIMING
			stopMmaSync = clock64();
			totalMmaSync += stopMmaSync - startMmaSync;
			startMmaStore = clock64();
#endif
			wmma::store_matrix_sync(outputTile + 32 * 16 + 16, c_frag, 32, wmma::mem_row_major);
#ifdef TIMING
			stopMmaStore = clock64();
			totalMmaStore += stopMmaStore - startMmaStore;
#endif
		}
		__syncthreads();

#ifdef TIMING
		stopMatmul = clock64();
		totalMatmul += stopMatmul - startMatmul;
#endif
		if (threadIdx.x < 32){
#ifdef DEBUG
		if (threadIdx.x == targetThreadIdx && blockIdx.x == targetBlockIdx && tileIndex == targetTileIndex){
			printf("TargetthreadIndex: %d, TargetBlockIndex: %d, TargetTileIndex: %d\n", targetThreadIdx, targetBlockIdx, targetTileIndex);
			printf("Output Tile 32x32:\n");
			for (int i = 0; i < 32; i++){
				for (int j = 0; j < 32; j++){
					printf("%f ", __half2float(outputTile[i * 32 + j]));
				}
				printf("\n");
			}
		}
#endif

			long int minIndex = 0;
			half minDistance = __habs(outputTile[threadIdx.x * 32]);
			half tempDistance;
			
			for (long int step = 1; step < 32; step++){
				tempDistance = __habs(outputTile[threadIdx.x * 32 + step]);
				if (tempDistance < minDistance){
					minDistance = tempDistance;
					minIndex = step;
				}
			}
#ifdef TIMING
			startStore = clock64();
#endif
			output[blockIdx.x * rowsPerBlock + tileIndex * 32 + threadIdx.x] = minIndex;
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
		printf("Kernel Time: %ld\n", totalKernel);
		printf("Load Time: %ld\n", totalLoad);
		printf("Store Time: %ld\n", totalStore);
		printf("Matmul Time: %ld\n", totalMatmul);
		printf("Sumsq Time: %ld\n", totalSumsq);
		printf("Mma Load Time: %ld\n", totalMmaLoad);
		printf("Mma Sync Time: %ld\n", totalMmaSync);
		printf("Mma Store Time: %ld\n", totalMmaStore);
		printf("\n");
	}
#endif
}

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

	for (long int run = 0; run < numRuns; run++) {
		cudaEventRecord(start);
		assign_labels<<<NBLOCKS, NTHREADS>>>(centroids, particles, output, dim, nParticles, nCentroids);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&milliseconds, start, end);
		totalSeconds += milliseconds/1000;
		milliseconds = 0.0f;

	}

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
    return 0;
}