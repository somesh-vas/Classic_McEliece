#include <iostream>
#include <cmath>
#include <chrono>

// GPU kernel for matrix multiplication
__global__ void matrixMulGPU(int* a, int* b, int* c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i = 0; i < size; ++i) {
            sum += a[row * size + i] * b[i * size + col];
        }
        c[row * size + col] = sum;
    }
}

// CPU function for matrix multiplication
void matrixMulCPU(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = sum;
        }
    }
}

int main() {
    // Matrix size
    int size = 2048;

    // Host matrices
    int* h_a = new int[size * size];
    int* h_b = new int[size * size];
    int* h_c_cpu = new int[size * size];
    int* h_c_gpu = new int[size * size];

    // Initialize host matrices
    for (int i = 0; i < size * size; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Device matrices
    int* d_a, *d_b, *d_c;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, size * size * sizeof(int));
    cudaMalloc((void**)&d_b, size * size * sizeof(int));
    cudaMalloc((void**)&d_c, size * size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

    // CPU matrix multiplication and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_a, h_b, h_c_cpu, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    // GPU matrix multiplication and timing
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);
    matrixMulGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, gpu_start, gpu_end);

    // Copy result from device to host
    cudaMemcpy(h_c_gpu, d_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for correctness (only for a small portion of the matrix due to the size)
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            int index = i * size + j;
            if (h_c_cpu[index] != h_c_gpu[index]) {
                std::cerr << "Error: CPU and GPU results do not match at index " << index << std::endl;
                break;
            }
        }
    }

    // Print timings
    std::cout << "Matrix Size: " << size << "x" << size << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpu_duration << " milliseconds" << std::endl;

    // Free allocated memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
