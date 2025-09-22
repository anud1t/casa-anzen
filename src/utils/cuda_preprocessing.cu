/**
 * @file cuda_preprocessing.cu
 * @brief CUDA-accelerated preprocessing implementation for Casa Anzen
 * @author Casa Anzen Team
 */

#include "utils/cuda_preprocessing.hpp"
#include "utils/cuda_utils.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace casa_anzen {

// CUDA kernel for BGR to RGB conversion and normalization
__global__ void bgr_to_rgb_kernel(const unsigned char* input, float* output,
                                 int width, int height, int src_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int src_idx = y * src_pitch + x * 3;
    int dst_idx = (y * width + x) * 3;
    
    // BGR to RGB conversion with normalization (0-255 -> 0-1)
    output[dst_idx + 0] = input[src_idx + 2] / 255.0f;  // R
    output[dst_idx + 1] = input[src_idx + 1] / 255.0f;  // G
    output[dst_idx + 2] = input[src_idx + 0] / 255.0f;  // B
}

// CUDA kernel for bilinear interpolation resize
__global__ void resize_kernel(const unsigned char* input, float* output,
                             int src_width, int src_height,
                             int dst_width, int dst_height,
                             int src_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Calculate source coordinates with bilinear interpolation
    float src_x = (x + 0.5f) * src_width / dst_width - 0.5f;
    float src_y = (y + 0.5f) * src_height / dst_height - 0.5f;
    
    int x1 = max(0, min(static_cast<int>(src_x), src_width - 1));
    int y1 = max(0, min(static_cast<int>(src_y), src_height - 1));
    int x2 = min(x1 + 1, src_width - 1);
    int y2 = min(y1 + 1, src_height - 1);
    
    float fx = src_x - x1;
    float fy = src_y - y1;
    
    // Bilinear interpolation weights
    float w11 = (1 - fx) * (1 - fy);
    float w12 = (1 - fx) * fy;
    float w21 = fx * (1 - fy);
    float w22 = fx * fy;
    
    int dst_idx = (y * dst_width + x) * 3;
    
    // Interpolate each channel
    for (int c = 0; c < 3; c++) {
        int src_idx1 = y1 * src_pitch + x1 * 3 + c;
        int src_idx2 = y1 * src_pitch + x2 * 3 + c;
        int src_idx3 = y2 * src_pitch + x1 * 3 + c;
        int src_idx4 = y2 * src_pitch + x2 * 3 + c;
        
        float val = w11 * input[src_idx1] + w12 * input[src_idx3] +
                   w21 * input[src_idx2] + w22 * input[src_idx4];
        
        output[dst_idx + c] = val / 255.0f;  // Normalize to 0-1
    }
}

// CUDA kernel for CHW format conversion
__global__ void hwc_to_chw_kernel(const float* input, float* output,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int hw_idx = y * width + x;
    int chw_idx_r = 0 * width * height + hw_idx;
    int chw_idx_g = 1 * width * height + hw_idx;
    int chw_idx_b = 2 * width * height + hw_idx;
    
    int input_idx = hw_idx * 3;
    output[chw_idx_r] = input[input_idx + 0];  // R
    output[chw_idx_g] = input[input_idx + 1];  // G
    output[chw_idx_b] = input[input_idx + 2];  // B
}

CUDAPreprocessor::CUDAPreprocessor(int input_width, int input_height)
    : input_width_(input_width)
    , input_height_(input_height)
    , buffer_size_(input_width * input_height * 3 * sizeof(float))
    , d_input_buffer_(nullptr)
    , d_resized_buffer_(nullptr)
    , d_normalized_buffer_(nullptr) {
}

CUDAPreprocessor::~CUDAPreprocessor() {
    freeBuffers();
}

void CUDAPreprocessor::allocateBuffers(cudaStream_t stream) {
    // Allocate input buffer (max size for typical camera resolutions)
    size_t max_input_size = 1920 * 1080 * 3;
    CUDA_CHECK(cudaMalloc(&d_input_buffer_, max_input_size));
    
    // Allocate intermediate buffers
    CUDA_CHECK(cudaMalloc(&d_resized_buffer_, buffer_size_));
    CUDA_CHECK(cudaMalloc(&d_normalized_buffer_, buffer_size_));
}

void CUDAPreprocessor::freeBuffers() {
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_resized_buffer_) {
        cudaFree(d_resized_buffer_);
        d_resized_buffer_ = nullptr;
    }
    if (d_normalized_buffer_) {
        cudaFree(d_normalized_buffer_);
        d_normalized_buffer_ = nullptr;
    }
}

void CUDAPreprocessor::preprocess(const cv::Mat& input_image, float* gpu_output, cudaStream_t stream) {
    int src_width = input_image.cols;
    int src_height = input_image.rows;
    int src_pitch = input_image.step;
    
    // Copy input image to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_input_buffer_, input_image.data, 
                               src_height * src_pitch, 
                               cudaMemcpyHostToDevice, stream));
    
    // Launch resize kernel
    dim3 block_size(16, 16);
    dim3 grid_size((input_width_ + block_size.x - 1) / block_size.x,
                   (input_height_ + block_size.y - 1) / block_size.y);
    
    resize_kernel<<<grid_size, block_size, 0, stream>>>(
        d_input_buffer_, d_resized_buffer_,
        src_width, src_height, input_width_, input_height_, src_pitch);
    
    // Launch HWC to CHW conversion kernel
    hwc_to_chw_kernel<<<grid_size, block_size, 0, stream>>>(
        d_resized_buffer_, gpu_output, input_width_, input_height_);
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace casa_anzen
