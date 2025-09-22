#pragma once

/**
 * @file cuda_postprocessing.hpp
 * @brief CUDA-accelerated postprocessing for YOLO outputs in Casa Anzen
 * @author Casa Anzen Team
 */

#include <cuda_runtime.h>
#include <vector>
#include "core/types.hpp"

namespace casa_anzen {

/**
 * @brief CUDA-accelerated postprocessing for YOLO inference
 * 
 * This class provides GPU-accelerated postprocessing operations including
 * NMS and detection filtering to eliminate CPU bottlenecks.
 */
class CUDAPostprocessor {
public:
    CUDAPostprocessor(int max_detections = 1000);
    ~CUDAPostprocessor();

    /**
     * @brief Process YOLO output on GPU
     * @param gpu_output Raw YOLO output from TensorRT
     * @param original_size Original image size
     * @param input_size Model input size
     * @param conf_threshold Confidence threshold
     * @param nms_threshold NMS threshold
     * @param stream CUDA stream
     * @return Vector of detections
     */
    std::vector<Detection> process(const float* gpu_output, 
                                  const cv::Size& original_size,
                                  const cv::Size& input_size,
                                  float conf_threshold,
                                  float nms_threshold,
                                  cudaStream_t stream,
                                  const std::vector<int>& target_classes = {},
                                  const cv::Rect& masked_area = cv::Rect());

    /**
     * @brief Allocate GPU buffers
     * @param output_size Size of YOLO output tensor
     * @param stream CUDA stream
     */
    void allocateBuffers(size_t output_size, cudaStream_t stream);

    /**
     * @brief Free GPU buffers
     */
    void freeBuffers();

private:
    int max_detections_;
    
    // GPU buffers
    float* d_output_buffer_;
    float* d_filtered_buffer_;
    int* d_valid_detections_;
    float* d_nms_buffer_;
    
    // Host buffers for results
    std::vector<float> h_filtered_buffer_;
    std::vector<int> h_valid_detections_;
};

} // namespace casa_anzen
