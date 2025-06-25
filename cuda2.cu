#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

// Filter kernels
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f };

float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f };

float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1 };

float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1 };

float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0 };

// Utility to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                        \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess)                                       \
        {                                                             \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__);     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

// Image structure
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

// Timer functions
double get_time() {
    static auto start = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start;
    return elapsed.count();
}

// Create a test image
void create_test_image(Image* image, int width, int height, int channels) {
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->data = (unsigned char*)malloc(width * height * channels);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                image->data[(y * width + x) * channels + c] =
                    (unsigned char)((x + y) % 256);
            }
        }
    }
}

// CPU convolution with clamping boundary handling
void convolutionCPU(const Image* input, Image* output, const float* filter, int filterWidth) {
    int halfWidth = filterWidth / 2;

    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            for (int c = 0; c < input->channels; c++) {
                float sum = 0.0f;

                for (int fy = 0; fy < filterWidth; fy++) {
                    for (int fx = 0; fx < filterWidth; fx++) {
                        int ix = x + fx - halfWidth;
                        int iy = y + fy - halfWidth;

                        // Clamp to image boundaries
                        ix = (ix < 0) ? 0 : ((ix >= input->width) ? input->width - 1 : ix);
                        iy = (iy < 0) ? 0 : ((iy >= input->height) ? input->height - 1 : iy);

                        float pixel = input->data[(iy * input->width + ix) * input->channels + c];
                        sum += pixel * filter[fy * filterWidth + fx];
                    }
                }

                // Clamp to 0-255 range
                sum = (sum < 0) ? 0 : ((sum > 255) ? 255 : sum);
                output->data[(y * input->width + x) * input->channels + c] = (unsigned char)sum;
            }
        }
    }
}

// Naive GPU kernel
__global__ void convolutionKernelNaive(unsigned char* input, unsigned char* output,
    const float* filter, int filterWidth,
    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfWidth = filterWidth / 2;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        for (int fy = 0; fy < filterWidth; fy++) {
            for (int fx = 0; fx < filterWidth; fx++) {
                int ix = x + fx - halfWidth;
                int iy = y + fy - halfWidth;

                // Clamp to image boundaries
                ix = (ix < 0) ? 0 : ((ix >= width) ? width - 1 : ix);
                iy = (iy < 0) ? 0 : ((iy >= height) ? height - 1 : iy);

                float pixel = input[(iy * width + ix) * channels + c];
                sum += pixel * filter[fy * filterWidth + fx];
            }
        }

        // Clamp to 0-255 range
        sum = (sum < 0) ? 0 : ((sum > 255) ? 255 : sum);
        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

// Shared memory kernel
__global__ void convolutionKernelShared(unsigned char* input, unsigned char* output,
    const float* filter, int filterWidth,
    int width, int height, int channels) {
    extern __shared__ unsigned char sharedData[];

    int blockX = blockIdx.x * blockDim.x;
    int blockY = blockIdx.y * blockDim.y;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    int x = blockX + threadX;
    int y = blockY + threadY;

    int halfWidth = filterWidth / 2;
    int tileWidth = blockDim.x + filterWidth - 1;
    int tileHeight = blockDim.y + filterWidth - 1;

    // Load data into shared memory
    for (int c = 0; c < channels; c++) {
        for (int ty = threadIdx.y; ty < tileHeight; ty += blockDim.y) {
            for (int tx = threadIdx.x; tx < tileWidth; tx += blockDim.x) {
                int ix = blockX + tx - halfWidth;
                int iy = blockY + ty - halfWidth;

                // Clamp to image boundaries
                ix = (ix < 0) ? 0 : ((ix >= width) ? width - 1 : ix);
                iy = (iy < 0) ? 0 : ((iy >= height) ? height - 1 : iy);

                sharedData[(ty * tileWidth + tx) * channels + c] = input[(iy * width + ix) * channels + c];
            }
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;

            for (int fy = 0; fy < filterWidth; fy++) {
                for (int fx = 0; fx < filterWidth; fx++) {
                    int tx = threadX + fx;
                    int ty = threadY + fy;

                    float pixel = sharedData[(ty * tileWidth + tx) * channels + c];
                    sum += pixel * filter[fy * filterWidth + fx];
                }
            }

            // Clamp to 0-255 range
            sum = (sum < 0) ? 0 : ((sum > 255) ? 255 : sum);
            output[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}

// Separable convolution kernels
__global__ void separableConvX(unsigned char* input, unsigned char* output,
    const float* filter, int filterWidth,
    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfWidth = filterWidth / 2;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        for (int fx = 0; fx < filterWidth; fx++) {
            int ix = x + fx - halfWidth;
            ix = (ix < 0) ? 0 : ((ix >= width) ? width - 1 : ix);

            float pixel = input[(y * width + ix) * channels + c];
            sum += pixel * filter[fx];
        }

        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

__global__ void separableConvY(unsigned char* input, unsigned char* output,
    const float* filter, int filterWidth,
    int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int halfWidth = filterWidth / 2;

    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        for (int fy = 0; fy < filterWidth; fy++) {
            int iy = y + fy - halfWidth;
            iy = (iy < 0) ? 0 : ((iy >= height) ? height - 1 : iy);

            float pixel = input[(iy * width + x) * channels + c];
            sum += pixel * filter[fy];
        }

        sum = (sum < 0) ? 0 : ((sum > 255) ? 255 : sum);
        output[(y * width + x) * channels + c] = (unsigned char)sum;
    }
}

// Compare images
bool compareImages(const Image* img1, const Image* img2) {
    if (img1->width != img2->width || img1->height != img2->height || img1->channels != img2->channels) {
        return false;
    }

    for (int i = 0; i < img1->width * img1->height * img1->channels; i++) {
        if (abs(img1->data[i] - img2->data[i]) > 1) { // Allow small differences due to floating point
            printf("Mismatch at %d: %d vs %d\n", i, img1->data[i], img2->data[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    // Image parameters
    int width = 1024;
    int height = 1024;
    int channels = 3;

    // Create test images
    Image inputImage;
    create_test_image(&inputImage, width, height, channels);

    Image outputCPU, outputNaive, outputShared, outputSeparable;
    outputCPU.width = outputNaive.width = outputShared.width = outputSeparable.width = width;
    outputCPU.height = outputNaive.height = outputShared.height = outputSeparable.height = height;
    outputCPU.channels = outputNaive.channels = outputShared.channels = outputSeparable.channels = channels;

    outputCPU.data = (unsigned char*)malloc(width * height * channels);
    outputNaive.data = (unsigned char*)malloc(width * height * channels);
    outputShared.data = (unsigned char*)malloc(width * height * channels);
    outputSeparable.data = (unsigned char*)malloc(width * height * channels);

    // Select filter
    float* filter = boxBlur3x3;
    int filterWidth = 3;

    // CPU convolution
    double start = get_time();
    convolutionCPU(&inputImage, &outputCPU, filter, filterWidth);
    double cpuTime = get_time() - start;
    printf("CPU time: %.3f ms\n", cpuTime * 1000);

    // Allocate GPU memory
    unsigned char* d_input, * d_output, * d_intermediate;
    float* d_filter;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, width * height * channels));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, width * height * channels));
    CHECK_CUDA_ERROR(cudaMalloc(&d_intermediate, width * height * channels));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, inputImage.data, width * height * channels, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

    // Naive GPU convolution
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    start = get_time();
    convolutionKernelNaive << <gridDim, blockDim >> > (d_input, d_output, d_filter, filterWidth, width, height, channels);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double naiveTime = get_time() - start;
    printf("Naive GPU time: %.3f ms\n", naiveTime * 1000);

    CHECK_CUDA_ERROR(cudaMemcpy(outputNaive.data, d_output, width * height * channels, cudaMemcpyDeviceToHost));

    // Verify naive implementation
    if (compareImages(&outputCPU, &outputNaive)) {
        printf("Naive GPU implementation matches CPU\n");
    }
    else {
        printf("Naive GPU implementation differs from CPU\n");
    }

    // Shared memory GPU convolution
    size_t sharedSize = (blockDim.x + filterWidth - 1) * (blockDim.y + filterWidth - 1) * channels;

    start = get_time();
    convolutionKernelShared << <gridDim, blockDim, sharedSize >> > (d_input, d_output, d_filter, filterWidth, width, height, channels);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double sharedTime = get_time() - start;
    printf("Shared memory GPU time: %.3f ms\n", sharedTime * 1000);

    CHECK_CUDA_ERROR(cudaMemcpy(outputShared.data, d_output, width * height * channels, cudaMemcpyDeviceToHost));

    // Verify shared memory implementation
    if (compareImages(&outputCPU, &outputShared)) {
        printf("Shared memory GPU implementation matches CPU\n");
    }
    else {
        printf("Shared memory GPU implementation differs from CPU\n");
    }

    // Separable convolution (for Gaussian blur)
    if (filter == gaussianBlur5x5) {
        float gaussian1D[5] = { 1 / 16.0f, 4 / 16.0f, 6 / 16.0f, 4 / 16.0f, 1 / 16.0f };
        float* d_gaussian1D;

        CHECK_CUDA_ERROR(cudaMalloc(&d_gaussian1D, 5 * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gaussian1D, gaussian1D, 5 * sizeof(float), cudaMemcpyHostToDevice));

        start = get_time();
        separableConvX << <gridDim, blockDim >> > (d_input, d_intermediate, d_gaussian1D, 5, width, height, channels);
        separableConvY << <gridDim, blockDim >> > (d_intermediate, d_output, d_gaussian1D, 5, width, height, channels);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        double separableTime = get_time() - start;
        printf("Separable GPU time: %.3f ms\n", separableTime * 1000);

        CHECK_CUDA_ERROR(cudaMemcpy(outputSeparable.data, d_output, width * height * channels, cudaMemcpyDeviceToHost));

        // Verify separable implementation
        if (compareImages(&outputCPU, &outputSeparable)) {
            printf("Separable GPU implementation matches CPU\n");
        }
        else {
            printf("Separable GPU implementation differs from CPU\n");
        }

        cudaFree(d_gaussian1D);
    }

    // Print speedup
    printf("Speedup (naive vs CPU): %.2fx\n", cpuTime / naiveTime);
    printf("Speedup (shared vs CPU): %.2fx\n", cpuTime / sharedTime);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate);
    cudaFree(d_filter);

    free(inputImage.data);
    free(outputCPU.data);
    free(outputNaive.data);
    free(outputShared.data);
    free(outputSeparable.data);

    return 0;
}