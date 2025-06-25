#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

const int WIDTH = 2048;
const int HEIGHT = 1024;
const int TILE_DIM = 32;
const float EPS = 1e-4f;

// --- CPU naive transpose ---
void transposeCPU(const float* in, float* out, int width, int height) {
    for (int row = 0; row < height; ++row)
        for (int col = 0; col < width; ++col)
            out[col * height + row] = in[row * width + col];
}

// --- GPU naive transpose ---
__global__ void transposeNaive(const float* in, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
        out[x * height + y] = in[y * width + x];
}

// --- GPU optimized (shared memory) ---
__global__ void transposeShared(float* in, float* out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

bool compare(const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i)
        if (std::fabs(a[i] - b[i]) > EPS)
            return false;
    return true;
}

int main() {
    int size = WIDTH * HEIGHT;
    size_t bytes = size * sizeof(float);

    float* in, * out_cpu, * out_gpu_naive, * out_gpu_opt;

    // --- Allocate unified memory
    CHECK_CUDA(cudaMallocManaged(&in, bytes));
    CHECK_CUDA(cudaMallocManaged(&out_gpu_naive, bytes));
    CHECK_CUDA(cudaMallocManaged(&out_gpu_opt, bytes));
    out_cpu = new float[bytes];

    // --- Fill with random data
    for (int i = 0; i < size; ++i)
        in[i] = static_cast<float>(rand()) / RAND_MAX;

    // --- CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    transposeCPU(in, out_cpu, WIDTH, HEIGHT);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

    // --- GPU naive
    dim3 block(32, 32);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    cudaEvent_t start_n, stop_n;
    cudaEventCreate(&start_n); cudaEventCreate(&stop_n);
    cudaEventRecord(start_n);
    transposeNaive << <grid, block >> > (in, out_gpu_naive, WIDTH, HEIGHT);
    cudaEventRecord(stop_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    float time_naive = 0;
    cudaEventElapsedTime(&time_naive, start_n, stop_n);

    // --- GPU optimized
    cudaEvent_t start_s, stop_s;
    cudaEventCreate(&start_s); cudaEventCreate(&stop_s);
    cudaEventRecord(start_s);
    transposeShared << <grid, block >> > (in, out_gpu_opt, WIDTH, HEIGHT);
    cudaEventRecord(stop_s);
    CHECK_CUDA(cudaDeviceSynchronize());
    float time_opt = 0;
    cudaEventElapsedTime(&time_opt, start_s, stop_s);

    // --- Verify correctness
    bool correct_naive = compare(out_cpu, out_gpu_naive, size);
    bool correct_opt = compare(out_cpu, out_gpu_opt, size);

    std::cout << "CPU time:           " << time_cpu << " ms\n";
    std::cout << "Naive GPU time:     " << time_naive << " ms\n";
    std::cout << "Optimized GPU time: " << time_opt << " ms\n";
    std::cout << "Naive correct:      " << (correct_naive ? "YES" : "NO") << "\n";
    std::cout << "Optimized correct:  " << (correct_opt ? "YES" : "NO") << "\n";

    // --- Clean up
    CHECK_CUDA(cudaFree(in));
    CHECK_CUDA(cudaFree(out_gpu_naive));
    CHECK_CUDA(cudaFree(out_gpu_opt));
    delete[] out_cpu;

    return 0;
}
