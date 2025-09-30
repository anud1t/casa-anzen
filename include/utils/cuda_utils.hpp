#pragma once

/**
 * @file cuda_utils.hpp
 * @brief CUDA utility functions and error handling for Casa Anzen
 * @author Casa Anzen Team
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace casa_anzen {

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << error << " \""                          \
                      << cudaGetErrorString(error) << "\"" << std::endl;     \
            throw std::runtime_error("CUDA error");                           \
        }                                                                      \
    } while (0)

// Get available GPU memory
inline void getGPUMemInfo(size_t& free_bytes, size_t& total_bytes) {
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

// Print GPU memory usage
inline void printGPUMemUsage() {
    size_t free_bytes, total_bytes;
    getGPUMemInfo(free_bytes, total_bytes);
    
    float free_mb = free_bytes / (1024.0f * 1024.0f);
    float total_mb = total_bytes / (1024.0f * 1024.0f);
    float used_mb = total_mb - free_mb;
    
    (void)used_mb; (void)total_mb; // Silence unused if logging disabled
}

// Get CUDA device properties
inline void printCUDADeviceInfo() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        (void)prop; // Suppress verbose device info in quiet mode
    }
}

} // namespace casa_anzen
