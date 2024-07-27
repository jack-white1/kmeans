#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

#define M 16777216
#define N 16
#define K 16

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // print the model of GPU we are using
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU model: " << prop.name << std::endl;

    cublasHandle_t handle;
    float *d_A, *d_B, *d_C;
    float *h_A, *h_B, *h_C;
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host memory
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];

    // Initialize host matrices
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    checkCudaStatus(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaStatus(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaStatus(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data from host to device
    checkCudaStatus(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaStatus(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    checkCublasStatus(cublasCreate(&handle));

    // Warmup run
    checkCublasStatus(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                   N, M, K,
                                   &alpha,
                                   d_B, CUDA_R_32F, N,
                                   d_A, CUDA_R_32F, K,
                                   &beta,
                                   d_C, CUDA_R_32F, N,
                                   CUBLAS_COMPUTE_32F_FAST_16F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Measure performance
    const int num_repeats = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_repeats; ++i) {
        checkCublasStatus(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                       N, M, K,
                                       &alpha,
                                       d_B, CUDA_R_32F, N,
                                       d_A, CUDA_R_32F, K,
                                       &beta,
                                       d_C, CUDA_R_32F, N,
                                       CUBLAS_COMPUTE_32F_FAST_16F,
                                       CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    checkCudaStatus(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() / num_repeats;

    // Calculate throughput
    double num_operations = (M * K + K * N);
    double throughput = (num_operations * sizeof(float)) / (avg_time * 1e9);  // GB/s


    std::cout << "Matrix multiplication (M = " << M << ", N = " << N << ", K = " << K << ", FP32 input, using TF32 tensor cores" << std::endl;
    std::cout << "\nTime for skinny matrix multiply using cuBLAS: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "Input data / execution time: " << throughput << " GB/s" << std::endl;

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}