/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef DEBUG_LOGGER_TENSOR_H
#define DEBUG_LOGGER_TENSOR_H

#include <iostream>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

#include <ctime>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

/**
 * @brief Debug utilities for tensor operations
 * 
 * This file contains specialized debug macros for tensor operations:
 * - DEBUG_DS_TENSOR: For DeepStream tensor debugging
 * - DEBUG_TENSOR: For general tensor debugging
 * 
 * These macros can be used independently or within a RAII debug logger scope.
 */

#define ENABLE_ENV_NAME "DEBUG"

/**
 * @brief Get formatted timestamp
 */
inline std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = std::localtime(&now_c);
    
    std::stringstream ss;
    ss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

/**
 * @brief Get header string with timestamp and file name
 */
inline std::string GetLogHeader() {
    return "[" + GetTimestamp() + "] " + std::string(__FILE__) + "] ";
}

/**
 * @brief Global variable to check if debug is enabled.
 * @note This is a static const variable that is initialized once and then reused.
 */
static const bool ENABLE_TENSOR_DEBUG = []() {
    const char* debug_env = getenv(ENABLE_ENV_NAME);
    if (!debug_env) {
        std::cout << "[" << __FILE__ << "] "
                  << "Environment variable " << ENABLE_ENV_NAME << " not set, DEBUG_DS_TENSOR disabled" 
                  << std::endl;
        return false;
    }
    bool is_enabled = (strcmp(debug_env, "1") == 0 || 
                      strcmp(debug_env, "true") == 0 || 
                      strcmp(debug_env, "TRUE") == 0);
    std::cout << "[" << __FILE__ << "] "
              << "Environment variable " << ENABLE_ENV_NAME << "=" << debug_env 
              << ", DEBUG_DS_TENSOR " << (is_enabled ? "enabled" : "disabled") 
              << std::endl;
    return is_enabled;
}();

/**
 * @brief Debug print with timestamp
 * @param ... Arguments to printf
 */
#define DEBUG_PRINT(...) \
    do { \
        if (ENABLE_TENSOR_DEBUG) { \
            std::cout << "[" << GetTimestamp() << "] DEBUG_PRINT: "; \
            printf(__VA_ARGS__); \
            std::cout << std::endl; \
        } \
    } while(0)

/**
 * @brief Calculate total elements from DS Infer dimensions
 * @param dims DS Infer dimensions to calculate total elements from
 * @return Total number of elements
 */
inline size_t GetTotalElements(const NvDsInferDims& dims) {
    size_t total = 1;
    for(unsigned int i = 0; i < dims.numDims; i++) {
        total *= dims.d[i];
    }
    return total;
}

/**
 * @brief Calculate coordinates for a block index
 * @param block_idx Block index to calculate coordinates for
 * @param dims DS Infer dimensions to calculate coordinates from
 * @param coords Vector to store coordinates
 */
inline void GetBlockCoordinates(size_t block_idx, const NvDsInferDims& dims, std::vector<size_t>& coords) {
    coords.clear();
    size_t remaining = block_idx;
    // Skip the last dimension as it represents the values within each block
    for(unsigned int d = 0; d < dims.numDims-1; d++) {  // Changed int to unsigned int
        coords.push_back(remaining % dims.d[d]);
        remaining /= dims.d[d];
    }
}

/**
 * @brief Format coordinates as string
 * @param coords Vector of coordinates to format
 * @return Formatted coordinates string
 */
inline std::string FormatCoordinates(const std::vector<size_t>& coords) {
    std::stringstream ss;
    ss << "[";
    for(size_t i = 0; i < coords.size(); i++) {
        ss << coords[i];
        if(i < coords.size()-1) ss << ",";
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief Debug print and dump for tensors
 * @param tensor_name Name of the tensor
 * @param data_type Data type of the tensor
 * @param layer_ref DS Infer layer reference
 * @param data Typed pointer to the tensor data
 * @param num_blocks Number of blocks in the tensor
 * @param last_dim_size Size of the last dimension
 * @note This function is used to dump the tensor data to a file.   
 * It prints the first 5 blocks of the tensor to the console and then dumps the full tensor to a file.
 *
 * @example
 * [2025-03-15 00:57:30] DEBUG: Entering NvDsInferParseCustomDDETRTAO
 * [2025-03-15 00:57:30] DEBUG: DS Tensor 'pred_boxes'
 * Dimensions (2D): [300, 4]
 * Block 0 [0]: 0.5762 0.5010 0.0685 0.0754
 * Block 1 [1]: 0.1716 0.3838 0.0967 0.0970
 * Block 2 [2]: 0.6582 0.5220 0.0612 0.0753
 * Block 3 [3]: 0.4597 0.4663 0.0687 0.0688
 * Block 4 [4]: 0.1991 0.5483 0.1510 0.2424
 * ... (truncated)
 * [2025-03-15 00:57:30] DEBUG: Full tensor dumped to /tmp/tensor_dump_pred_boxes_1742000250.txt
 * [2025-03-15 00:57:30] DEBUG: DS Tensor 'pred_logits'
 * Dimensions (2D): [300, 5]
 * Block 0 [0]: -7.3555 -4.0586 2.3086 -3.8867 -4.6758
 * Block 1 [1]: -7.1875 -4.2344 2.5039 -4.1758 -3.6934
 * Block 2 [2]: -6.8984 -4.0039 1.7812 -3.5215 -4.7070
 * Block 3 [3]: -7.3516 -4.2617 2.3906 -3.4824 -4.5156
 * Block 4 [4]: -7.3906 -3.9648 2.1016 -3.6211 -3.6543
 * ... (truncated)
 * [2025-03-15 00:57:30] DEBUG: Full tensor dumped to /tmp/tensor_dump_pred_logits_1742000250.txt
 *
 * For 3D tensor [8, 4, 2]:
 * [2024-03-14 15:30:45] DEBUG: DS Tensor 'example'
 * Dimensions (3D): [8, 4, 2]
 * Block 0 [0,0]: 0.1 0.2
 * Block 1 [0,1]: 0.3 0.4
 * Block 2 [0,2]: 0.5 0.6
 * Block 3 [0,3]: 0.7 0.8
 * Block 4 [1,0]: 0.9 1.0
 * ...
 */
template<typename T>
void DumpTensorData(const char* tensor_name, const char* data_type,
                   const NvDsInferLayerInfo& layer_ref,
                   const T* data, size_t num_blocks, size_t last_dim_size) {
    const size_t blocks_to_show = std::min((size_t)5, num_blocks);
    std::vector<size_t> coords;

    // Print preview to console
    for(size_t block = 0; block < blocks_to_show; block++) {
        // Calculate and show coordinates for this block
        GetBlockCoordinates(block, layer_ref.inferDims, coords);
        std::cout << "Block " << block << " " << FormatCoordinates(coords) << ": ";

        // Print the actual values
        for(size_t i = 0; i < last_dim_size; i++) {
            std::cout << data[block * last_dim_size + i] << " ";
        }
        std::cout << std::endl;
    }
    if(num_blocks > blocks_to_show) std::cout << "... (truncated)\n";

    // Dump to file
    std::string filename = std::string("/tmp/tensor_dump_") + tensor_name + "_" +
                          std::to_string(std::time(nullptr)) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "ERROR: Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Set higher precision for file output
    outfile.precision(6);
    outfile << std::fixed;

    outfile << "Tensor name: " << tensor_name << "\n";
    outfile << "Data type: " << data_type << "\n";
    outfile << "Dimensions (" << layer_ref.inferDims.numDims << "D): [";
    for(unsigned int i = 0; i < layer_ref.inferDims.numDims; i++) {
        outfile << layer_ref.inferDims.d[i];
        if(i < layer_ref.inferDims.numDims-1) outfile << ", ";
    }
    outfile << "]\n\n";

    // Write all blocks with coordinates
    for(size_t block = 0; block < num_blocks; block++) {
        GetBlockCoordinates(block, layer_ref.inferDims, coords);
        outfile << "Block " << block << " " << FormatCoordinates(coords) << ":\n";
        for(size_t i = 0; i < last_dim_size; i++) {
            outfile << data[block * last_dim_size + i] << " ";
            if((i + 1) % 10 == 0) outfile << "\n";
        }
        outfile << "\n\n";
    }
    outfile.close();
    std::cout << "[" << GetTimestamp() << "] DEBUG_DS_TENSOR: Full tensor dumped to " << filename << std::endl;
}

/**
 * @brief Debug print and dump for DS Infer tensors
 * @param layer_arg DS Infer layer reference
 * @note This macro is used to print and dump the tensor data for DS Infer.
 * It validates the tensor, prints the dimensions, and then prints the tensor data based on the data type.
 */
#define DEBUG_DS_TENSOR(layer_arg) \
    do { \
        if (ENABLE_TENSOR_DEBUG) { \
            const NvDsInferLayerInfo& layer_ref = layer_arg; \
            const char* tensor_name = layer_ref.layerName; \
            std::cout << "[" << GetTimestamp() << "] DEBUG_DS_TENSOR: '" << tensor_name << "'\n"; \
            \
            /* Validate tensor */ \
            if (!layer_ref.buffer) { \
                std::cerr << "ERROR: Null tensor buffer\n"; \
                break; \
            } \
            if (layer_ref.inferDims.numDims == 0) { \
                std::cerr << "ERROR: Invalid tensor dimensions\n"; \
                break; \
            } \
            \
            /* Print dimensions */ \
            std::cout << "Dimensions (" << layer_ref.inferDims.numDims << "D): ["; \
            for(unsigned int i = 0; i < layer_ref.inferDims.numDims; i++) { \
                std::cout << layer_ref.inferDims.d[i]; \
                if(i < layer_ref.inferDims.numDims-1) std::cout << ", "; \
            } \
            std::cout << "]\n"; \
            \
            /* Calculate layout */ \
            const size_t total_elements = GetTotalElements(layer_ref.inferDims); \
            const size_t last_dim_size = layer_ref.inferDims.d[layer_ref.inferDims.numDims-1]; \
            const size_t num_blocks = total_elements / last_dim_size; \
            \
            /* Set output precision for floating point */ \
            std::cout.precision(4); \
            std::cout << std::fixed; \
            \
            /* Print tensor data based on type */ \
            switch(layer_ref.dataType) { \
                case FLOAT: { \
                    const float* data = static_cast<const float*>(layer_ref.buffer); \
                    DumpTensorData(tensor_name, "FLOAT", layer_ref, data, num_blocks, last_dim_size); \
                    break; \
                } \
                case INT32: { \
                    const int32_t* data = static_cast<const int32_t*>(layer_ref.buffer); \
                    DumpTensorData(tensor_name, "INT32", layer_ref, data, num_blocks, last_dim_size); \
                    break; \
                } \
                case INT64: { \
                    const int64_t* data = static_cast<const int64_t*>(layer_ref.buffer); \
                    DumpTensorData(tensor_name, "INT64", layer_ref, data, num_blocks, last_dim_size); \
                    break; \
                } \
                default: \
                    std::cerr << "ERROR: Unsupported data type: " << layer_ref.dataType << std::endl; \
            } \
        } \
    } while(0)

/**
 * @brief Debug print and dump for generic tensors
 * @param tensor_name Name of the tensor
 * @param data Pointer to the tensor data
 * @param size Size of the tensor
 * @param type Data type of the tensor
 * 
 * @note This macro is used to print and dump the tensor data.
 * It prints the first 100 values of the tensor to the console and then dumps the full tensor to a file.
 */
#define DEBUG_TENSOR(tensor_name, data, size, type) \
    if (ENABLE_TENSOR_DEBUG) { \
        /* Console print first 100 values */ \
        std::cout << "[" << GetTimestamp() << "] DEBUG_TENSOR: '" << tensor_name \
                  << "' (showing first 100 of " << size << " values):" << std::endl; \
        const type* typed_data = static_cast<const type*>(data); \
        for (size_t i = 0; i < size && i < 100; i++) { \
            std::cout << typed_data[i] << " "; \
            if ((i + 1) % 10 == 0) std::cout << std::endl; \
        } \
        if (size > 100) std::cout << "... (truncated)" << std::endl; \
        std::cout << std::endl; \
        /* Dump complete tensor to file */ \
        std::string filename = std::string("/tmp/tensor_dump_") + tensor_name + "_" + \
                              std::to_string(std::time(nullptr)) + ".txt"; \
        std::ofstream outfile(filename); \
        if (outfile.is_open()) { \
            outfile << "Tensor name: " << tensor_name << "\n"; \
            outfile << "Data type: " << #type << "\n"; \
            outfile << "Size: " << size << "\n"; \
            outfile << "Values:\n"; \
            for (size_t i = 0; i < size; i++) { \
                outfile << typed_data[i] << " "; \
                if ((i + 1) % 10 == 0) outfile << "\n"; \
            } \
            outfile.close(); \
            std::cout << "[" << GetTimestamp() << "] DEBUG_TENSOR: Full tensor dumped to " << filename << std::endl; \
        } \
    }

#endif // DEBUG_LOGGER_TENSOR_H