// matmul.cu — DATA 266 HW1, Task 3 (Rohil, SID4=2708)
// Square matrix multiply C = A * B on CPU (baseline) and GPU (CUDA),
// timing CPU compute, GPU kernel, and H2D+D2H transfers for N = 256, 1024, 4096.
//
// Build (Colab GPU):   !nvcc -O3 -arch=sm_75 matmul.cu -o matmul
// Run:                 !./matmul
//
// -------------------- BLOCKS AND THREADS --------------------
// The output matrix C is N x N. We map ONE CUDA THREAD to ONE output element C[row][col].
// Threads are grouped into 2-D BLOCKS of BLOCK x BLOCK threads (here 16x16 = 256 threads/block).
// Blocks are arranged in a 2-D GRID that tiles the whole N x N output:
//     grid.x = ceil(N / BLOCK)   covers the columns
//     grid.y = ceil(N / BLOCK)   covers the rows
// Inside the kernel each thread recovers its global coordinates from its block and thread indices:
//     col = blockIdx.x * blockDim.x + threadIdx.x
//     row = blockIdx.y * blockDim.y + threadIdx.y
// Because N may not divide evenly by BLOCK, we launch a few extra threads and guard with
// `if (row < N && col < N)` so out-of-range threads do nothing.
// ------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK 16   // 16 x 16 = 256 threads per block

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// GPU kernel: each thread computes one element of C.
__global__ void matmulKernel(const float* A, const float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // which column of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // which row of C
    if (row < N && col < N) {                          // guard ragged edge
        float acc = 0.0f;
        for (int k = 0; k < N; ++k)
            acc += A[row * N + k] * B[k * N + col];    // dot product of A-row and B-col
        C[row * N + col] = acc;
    }
}

// Naive CPU baseline (triple loop). O(N^3); intentionally simple for a fair "no-GPU" reference.
void matmulCPU(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k)
                acc += A[i * N + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

void fillRandom(float* M, int n) {
    for (int i = 0; i < n; ++i) M[i] = (float)rand() / (float)RAND_MAX;
}

void benchmark(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes);
    float *hC_cpu = (float*)malloc(bytes), *hC_gpu = (float*)malloc(bytes);
    srand(2708);                          // SEED for reproducible inputs
    fillRandom(hA, N * N); fillRandom(hB, N * N);

    // ---------- CPU baseline ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(hA, hB, hC_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU ----------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t eH2D0, eH2D1, eK0, eK1, eD2H0, eD2H1;
    cudaEventCreate(&eH2D0); cudaEventCreate(&eH2D1);
    cudaEventCreate(&eK0);   cudaEventCreate(&eK1);
    cudaEventCreate(&eD2H0); cudaEventCreate(&eD2H1);

    // Host -> Device
    cudaEventRecord(eH2D0);
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(eH2D1);

    // Launch configuration: 2-D grid of 16x16 blocks tiling the N x N output
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    // Warm-up launch (excluded from timing) so first-call overhead doesn't skew the kernel time
    matmulKernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed kernel
    cudaEventRecord(eK0);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(eK1);
    CUDA_CHECK(cudaGetLastError());

    // Device -> Host
    cudaEventRecord(eD2H0);
    CUDA_CHECK(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2H1);
    CUDA_CHECK(cudaEventSynchronize(eD2H1));

    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    cudaEventElapsedTime(&h2d_ms, eH2D0, eH2D1);
    cudaEventElapsedTime(&kernel_ms, eK0, eK1);
    cudaEventElapsedTime(&d2h_ms, eD2H0, eD2H1);
    float transfer_ms = h2d_ms + d2h_ms;
    float e2e_ms = h2d_ms + kernel_ms + d2h_ms;   // GPU end-to-end (transfers + kernel)

    // ---------- correctness check (max abs diff vs CPU) ----------
    double maxdiff = 0.0;
    for (int i = 0; i < N * N; ++i)
        maxdiff = fmax(maxdiff, fabs((double)hC_cpu[i] - (double)hC_gpu[i]));

    printf("N=%-5d | CPU %10.2f ms | kernel %8.3f ms | H2D+D2H %8.3f ms | "
           "e2e %9.3f ms | speedup(CPU/e2e) %6.2fx | maxdiff %.2e\n",
           N, cpu_ms, kernel_ms, transfer_ms, e2e_ms, cpu_ms / e2e_ms, maxdiff);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC_cpu); free(hC_gpu);
}

int main() {
    int sizes[] = {256, 1024, 4096};
    printf("Square matmul CPU vs CUDA (BLOCK=%dx%d = %d threads/block)\n",
           BLOCK, BLOCK, BLOCK * BLOCK);
    for (int s : sizes) benchmark(s);
    return 0;
}
