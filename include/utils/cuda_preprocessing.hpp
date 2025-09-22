#pragma once

/**
 * @file cuda_preprocessing.hpp
 * @brief CUDA-accelerated image preprocessing for Casa Anzen
 * @author Casa Anzen Team
 */

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <vector>

namespace casa_anzen {

/**
 * @brief CUDA-accelerated image preprocessing for YOLO inference
 * 
 * This class provides GPU-accelerated preprocessing operations to eliminate
 * CPU bottlenecks in the inference pipeline.
 */
class CUDAPreprocessor {
public:
    CUDAPreprocessor(int input_width, int input_height);
    ~CUDAPreprocessor();

    /**
     * @brief Preprocess image directly on GPU
     * @param input_image Input image (BGR format)
     * @param gpu_output Pre-allocated GPU buffer for output
     * @param stream CUDA stream for asynchronous execution
     */
    void preprocess(const cv::Mat& input_image, float* gpu_output, cudaStream_t stream);

    /**
     * @brief Get required GPU buffer size
     * @return Size in bytes
     */
    size_t getBufferSize() const { return buffer_size_; }

    /**
     * @brief Allocate GPU buffers
     * @param stream CUDA stream
     */
    void allocateBuffers(cudaStream_t stream);

    /**
     * @brief Free GPU buffers
     */
    void freeBuffers();

private:
    int input_width_;
    int input_height_;
    size_t buffer_size_;
    
    // GPU buffers for intermediate processing
    unsigned char* d_input_buffer_;
    float* d_resized_buffer_;
    float* d_normalized_buffer_;
    
    // Pre-computed scaling factors
    float scale_x_;
    float scale_y_;
};

} // namespace casa_anzen
