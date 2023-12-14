#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

typedef unsigned short gf; // Assuming gf is a 16-bit unsigned type

// CPU implementation of bitrev
gf bitrev_cpu(gf a) {
    a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8);
    a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4);
    a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2);
    a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1);
    return a >> 4;
}

__device__ gf bitrev_gpu(gf a) {
    a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8);
    a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4);
    a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2);
    a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1);
    return a >> 4;
}

__global__ void bitrev_kernel(gf *in, gf *out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Device version of bitrev_cpu
        gf a = in[tid];
        a = ((a & 0x00FF) << 8) | ((a & 0xFF00) >> 8);
        a = ((a & 0x0F0F) << 4) | ((a & 0xF0F0) >> 4);
        a = ((a & 0x3333) << 2) | ((a & 0xCCCC) >> 2);
        a = ((a & 0x5555) << 1) | ((a & 0xAAAA) >> 1);
        out[tid] = a >> 4;
    }
}

int main() {
    const int size = 1000000; // Adjust the size based on your testing
    const int dataSize = size * sizeof(gf);

    gf *h_in = (gf*)malloc(dataSize);
    gf *h_out_cpu = (gf*)malloc(dataSize);
    gf *h_out_gpu = (gf*)malloc(dataSize);
    gf *d_in, *d_out;

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_in[i] = i;
    }

    // CPU computation with timing
    clock_t start_cpu, stop_cpu;
    start_cpu = clock();
    for (int i = 0; i < size; i++) {
        h_out_cpu[i] = bitrev_cpu(h_in[i]);
    }
    stop_cpu = clock();

    printf("CPU Computation Time: %f ms\n", ((double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC) * 1000.0);

    // Allocate device memory
    cudaMalloc((void**)&d_in, dataSize);
    cudaMalloc((void**)&d_out, dataSize);

    // Copy input data from host to device with timing
    cudaMemcpy(d_in, h_in, dataSize, cudaMemcpyHostToDevice);

    // GPU computation with timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    // Launch the kernel with one block and one thread per element
    bitrev_kernel<<<(size + 255) / 256, 256>>>(d_in, d_out, size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    // Copy the result back to the host
    cudaMemcpy(h_out_gpu, d_out, dataSize, cudaMemcpyDeviceToHost);

    // Print the results (optional)
    // for (int i = 0; i < size; i++) {
    //     printf("a: %04x, bitrev_gpu(a): %04x\n", h_in[i], h_out_gpu[i]);
    // }

    // Calculate GPU operation cycle counts
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);

    printf("GPU Computation Time: %f ms\n", gpu_time);

    // Free memory and CUDA events
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
