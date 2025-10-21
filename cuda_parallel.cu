#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// --- KERNEL CODE (Device) ---
// This kernel performs the parallel reduction within a single block.
__global__ void reduceSum(int *g_input, int *g_output) {
    // Dynamic shared memory allocation
    extern __shared__ int s_data[]; 
    
    // Global index: determines the element in the input array this thread reads
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // Local index: the thread's position within its block (0 to blockDim.x - 1)
    unsigned int tid = threadIdx.x; 
    
    // Load data from global memory to shared memory
    s_data[tid] = g_input[i];
    
    // Synchronize to ensure all data is loaded before starting the reduction
    __syncthreads();
    
    // Logarithmic reduction (tree reduction)
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Synchronize after each step to ensure all additions are complete
        __syncthreads();
    }
    
    // The final sum for the block is in s_data[0]. 
    // Thread 0 writes the result to global memory.
    if (tid == 0) {
        g_output[blockIdx.x] = s_data[0];
    }
}

// --- HOST CODE (CPU) ---
int main() {
    const int N = 1024 * 1024; // Total number of elements
    const int BLOCK_SIZE = 512;  // Threads per block
    
    // Calculate Grid Size
    // Each block will process BLOCK_SIZE elements and produce one partial sum.
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Size in bytes
    const size_t arraySize = N * sizeof(int);
    const size_t outputSize = NUM_BLOCKS * sizeof(int);

    // Host memory allocation
    int *h_input = (int*)malloc(arraySize);
    
    // Initialize host input data (e.g., set all elements to 1)
    long long expectedSum = 0;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
        expectedSum += h_input[i];
    }
    
    // Device memory pointers
    int *d_input, *d_output;

    // 1. Memory Allocation on Device
    CUDA_CHECK(cudaMalloc((void**)&d_input, arraySize));
    CUDA_CHECK(cudaMalloc((void**)&d_output, outputSize));
    
    // 2. Data Transfer: Host to Device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, arraySize, cudaMemcpyHostToDevice));

    // 3. Kernel Launch (First Pass: Partial Sums)
    // The shared memory size is BLOCK_SIZE * sizeof(int)
    reduceSum<<<NUM_BLOCKS, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_input, d_output);
    
    // Synchronize to wait for all blocks to finish their partial sums
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 4. Data Transfer: Device to Host (Partial Sums)
    // We reuse h_input for simplicity to store the partial sums
    int *h_output = (int*)malloc(outputSize);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // --- FINAL REDUCTION ON CPU (Second Pass) ---
    // For simplicity, we sum the partial results on the CPU.
    // For larger problems, a second, smaller kernel is often used.
    long long finalSum = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        finalSum += h_output[i];
    }

    // 5. Verification and Cleanup
    printf("Array Size (N): %d\n", N);
    printf("Blocks Launched: %d\n", NUM_BLOCKS);
    printf("Expected Sum: %lld\n", expectedSum);
    printf("CUDA Final Sum: %lld\n", finalSum);

    if (finalSum == expectedSum) {
        printf("Result: SUCCESS! 🎉\n");
    } else {
        printf("Result: FAILURE! 😔\n");
    }

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
