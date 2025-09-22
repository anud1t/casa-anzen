/**
 * @file cuda_postprocessing.cu
 * @brief CUDA-accelerated postprocessing implementation for Casa Anzen
 * @author Casa Anzen Team
 */

#include "utils/cuda_postprocessing.hpp"
#include "utils/cuda_utils.hpp"
#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace casa_anzen {

// CUDA kernel for confidence filtering
__global__ void confidence_filter_kernel(const float* input, float* output,
                                        int* valid_count, int num_detections,
                                        int num_classes, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_detections) return;
    
    // Find best class score for this detection
    float best_score = 0.0f;
    int best_class = 0;
    
    // Start from index 4 (after x, y, w, h)
    for (int c = 0; c < num_classes; c++) {
        float score = input[(4 + c) * num_detections + idx];
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    
    // Filter by confidence threshold
    if (best_score >= threshold) {
        int output_idx = atomicAdd(valid_count, 1);
        
        // Copy detection data: [x, y, w, h, class_id, confidence]
        output[output_idx * 6 + 0] = input[0 * num_detections + idx]; // x
        output[output_idx * 6 + 1] = input[1 * num_detections + idx]; // y
        output[output_idx * 6 + 2] = input[2 * num_detections + idx]; // w
        output[output_idx * 6 + 3] = input[3 * num_detections + idx]; // h
        output[output_idx * 6 + 4] = static_cast<float>(best_class);  // class_id
        output[output_idx * 6 + 5] = best_score;                     // confidence
    }
}

// CUDA kernel for coordinate transformation
__global__ void coordinate_transform_kernel(float* detections,
                                           int num_detections,
                                           float scale_x, float scale_y,
                                           int original_width, int original_height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_detections) return;
    
    int base_idx = idx * 6;
    
    // Get center coordinates and dimensions
    float cx = detections[base_idx + 0];
    float cy = detections[base_idx + 1];
    float w = detections[base_idx + 2];
    float h = detections[base_idx + 3];
    
    // Convert to corner coordinates
    float x1 = (cx - w/2) * scale_x;
    float y1 = (cy - h/2) * scale_y;
    float x2 = (cx + w/2) * scale_x;
    float y2 = (cy + h/2) * scale_y;
    
    // Clamp to image bounds
    x1 = max(0.0f, min(x1, static_cast<float>(original_width - 1)));
    y1 = max(0.0f, min(y1, static_cast<float>(original_height - 1)));
    x2 = max(0.0f, min(x2, static_cast<float>(original_width - 1)));
    y2 = max(0.0f, min(y2, static_cast<float>(original_height - 1)));
    
    // Store transformed coordinates
    detections[base_idx + 0] = x1; // x1
    detections[base_idx + 1] = y1; // y1
    detections[base_idx + 2] = x2 - x1; // width
    detections[base_idx + 3] = y2 - y1; // height
}

// CUDA kernel for NMS (simplified version)
__global__ void nms_kernel(const float* detections, float* output,
                          int* valid_count, int num_detections,
                          float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_detections) return;
    
    int base_idx = idx * 6;
    float x1 = detections[base_idx + 0];
    float y1 = detections[base_idx + 1];
    float w1 = detections[base_idx + 2];
    float h1 = detections[base_idx + 3];
    float conf1 = detections[base_idx + 5];
    
    bool should_keep = true;
    
    // Check IoU with all other detections
    for (int j = 0; j < num_detections; j++) {
        if (j == idx) continue;
        
        int j_base_idx = j * 6;
        float x2 = detections[j_base_idx + 0];
        float y2 = detections[j_base_idx + 1];
        float w2 = detections[j_base_idx + 2];
        float h2 = detections[j_base_idx + 3];
        float conf2 = detections[j_base_idx + 5];
        
        // Skip if this detection has lower confidence
        if (conf2 <= conf1) continue;
        
        // Calculate IoU
        float x_left = max(x1, x2);
        float y_top = max(y1, y2);
        float x_right = min(x1 + w1, x2 + w2);
        float y_bottom = min(y1 + h1, y2 + h2);
        
        if (x_right > x_left && y_bottom > y_top) {
            float intersection = (x_right - x_left) * (y_bottom - y_top);
            float area1 = w1 * h1;
            float area2 = w2 * h2;
            float union_area = area1 + area2 - intersection;
            float iou = intersection / union_area;
            
            if (iou > threshold) {
                should_keep = false;
                break;
            }
        }
    }
    
    if (should_keep) {
        int output_idx = atomicAdd(valid_count, 1);
        int output_base_idx = output_idx * 6;
        
        // Copy detection to output
        for (int k = 0; k < 6; k++) {
            output[output_base_idx + k] = detections[base_idx + k];
        }
    }
}

CUDAPostprocessor::CUDAPostprocessor(int max_detections)
    : max_detections_(max_detections)
    , d_output_buffer_(nullptr)
    , d_filtered_buffer_(nullptr)
    , d_valid_detections_(nullptr)
    , d_nms_buffer_(nullptr) {
    
    // Pre-allocate host buffers
    h_filtered_buffer_.resize(max_detections * 6);
    h_valid_detections_.resize(1);
}

CUDAPostprocessor::~CUDAPostprocessor() {
    freeBuffers();
}

void CUDAPostprocessor::allocateBuffers(size_t output_size, cudaStream_t stream) {
    // Allocate GPU buffers
    CUDA_CHECK(cudaMalloc(&d_output_buffer_, output_size));
    CUDA_CHECK(cudaMalloc(&d_filtered_buffer_, max_detections_ * 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_detections_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nms_buffer_, max_detections_ * 6 * sizeof(float)));
}

void CUDAPostprocessor::freeBuffers() {
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    if (d_filtered_buffer_) {
        cudaFree(d_filtered_buffer_);
        d_filtered_buffer_ = nullptr;
    }
    if (d_valid_detections_) {
        cudaFree(d_valid_detections_);
        d_valid_detections_ = nullptr;
    }
    if (d_nms_buffer_) {
        cudaFree(d_nms_buffer_);
        d_nms_buffer_ = nullptr;
    }
}

std::vector<Detection> CUDAPostprocessor::process(const float* gpu_output, 
                                                 const cv::Size& original_size,
                                                 const cv::Size& input_size,
                                                 float conf_threshold,
                                                 float nms_threshold,
                                                 cudaStream_t stream,
                                                 const std::vector<int>& target_classes,
                                                 const cv::Rect& masked_area) {
    std::vector<Detection> detections;
    
    // Calculate scale factors
    float scale_x = static_cast<float>(original_size.width) / input_size.width;
    float scale_y = static_cast<float>(original_size.height) / input_size.height;
    
    // YOLOv11n parameters
    int num_detections = 8400;
    int num_classes = 79;
    
    // Reset valid count
    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(d_valid_detections_, &zero, sizeof(int), 
                               cudaMemcpyHostToDevice, stream));
    
    // Step 1: Confidence filtering
    dim3 block_size(256);
    dim3 grid_size((num_detections + block_size.x - 1) / block_size.x);
    
    confidence_filter_kernel<<<grid_size, block_size, 0, stream>>>(
        gpu_output, d_filtered_buffer_, d_valid_detections_,
        num_detections, num_classes, conf_threshold);
    
    // Get number of valid detections
    int valid_count;
    CUDA_CHECK(cudaMemcpyAsync(&valid_count, d_valid_detections_, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    
    if (valid_count == 0) {
        return detections;
    }
    
    // Step 2: Coordinate transformation
    coordinate_transform_kernel<<<grid_size, block_size, 0, stream>>>(
        d_filtered_buffer_, valid_count, scale_x, scale_y,
        original_size.width, original_size.height);
    
    // Step 3: NMS (if needed)
    if (nms_threshold > 0.0f) {
        // Reset valid count for NMS
        CUDA_CHECK(cudaMemcpyAsync(d_valid_detections_, &zero, sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        
        nms_kernel<<<grid_size, block_size, 0, stream>>>(
            d_filtered_buffer_, d_nms_buffer_, d_valid_detections_,
            valid_count, nms_threshold);
        
        // Get final count after NMS
        CUDA_CHECK(cudaMemcpyAsync(&valid_count, d_valid_detections_, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        
        // Copy final results to host
        if (valid_count > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_filtered_buffer_.data(), d_nms_buffer_,
                                       valid_count * 6 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
    } else {
        // Copy filtered results to host
        CUDA_CHECK(cudaMemcpyAsync(h_filtered_buffer_.data(), d_filtered_buffer_,
                                   valid_count * 6 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    cudaStreamSynchronize(stream);
    
    // Convert to Detection objects
    detections.reserve(valid_count);
    for (int i = 0; i < valid_count; i++) {
        int base_idx = i * 6;
        
        Detection detection;
        detection.bbox = cv::Rect(
            static_cast<int>(h_filtered_buffer_[base_idx + 0]), // x
            static_cast<int>(h_filtered_buffer_[base_idx + 1]), // y
            static_cast<int>(h_filtered_buffer_[base_idx + 2]), // width
            static_cast<int>(h_filtered_buffer_[base_idx + 3])  // height
        );
        detection.confidence = h_filtered_buffer_[base_idx + 5];
        detection.class_id = static_cast<int>(h_filtered_buffer_[base_idx + 4]);
        
        // Map class IDs to correct labels
        static const std::vector<std::string> COCO_CLASSES = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        };
        
        if (detection.class_id >= 0 && detection.class_id < COCO_CLASSES.size()) {
            detection.class_name = COCO_CLASSES[detection.class_id];
        } else {
            detection.class_name = "unknown";
        }
        
        // Apply class filter if specified
        if (!target_classes.empty()) {
            if (std::find(target_classes.begin(), target_classes.end(), detection.class_id) == target_classes.end()) {
                continue; // Skip this detection
            }
        }
        
        // Apply masked area filter if specified
        if (masked_area.area() > 0) {
            cv::Rect detection_rect = detection.bbox;
            cv::Rect intersection = detection_rect & masked_area;
            if (intersection.area() > 0) {
                continue; // Skip detections in masked area
            }
        }
        
        detections.push_back(detection);
    }
    
    return detections;
}

} // namespace casa_anzen
