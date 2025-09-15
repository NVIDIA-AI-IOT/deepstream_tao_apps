/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>

// DEBUG print includes
#include "debug_logger_raii.hpp"
#include "debug_logger_tensor.hpp"
// end DEBUG print includes

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)


struct MrcnnRawDetection {
    float y1, x1, y2, x2, class_id, score;
};
/* This is a sample bounding box parsing function for the sample FasterRCNN
 *
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferInitializeInputLayers (std::vector<NvDsInferLayerInfo> const &inputLayersInfo,
        NvDsInferNetworkInfo const &networkInfo,
        unsigned int maxBatchSize);

extern "C"
bool NvDsInferParseCustomNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomBatchedYoloV5NMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomEfficientDetTAO (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomSegformerTAO(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap);

extern "C"
bool NvDsInferParseCustomChangeNet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap);

extern "C"
bool NvDsInferParseCustomDDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferClassiferParseNonSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);
/** Implementation of a custom processor for DeepStream nvinfer
 * plugin to process additonal inputs.
 */
extern "C" {

bool NvDsInferInitializeInputLayers (std::vector<NvDsInferLayerInfo> const &inputLayersInfo,
        NvDsInferNetworkInfo const &networkInfo,
        unsigned int maxBatchSize)
{
   /* Nothing to do, no input layers with static values are expected*/
  return true;
}
}

extern "C"
bool NvDsInferParseCustomNMSTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 2)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 2 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    // Host memory for "nms" which has 2 output bindings:
    // the order is bboxes and keep_count
    float* out_nms = (float *) outputLayersInfo[0].buffer;
    int * p_keep_count = (int *) outputLayersInfo[1].buffer;
    const int out_class_size = detectionParams.numClassesConfigured;
    const float threshold = detectionParams.perClassThreshold[0];

    float* det;

    for (int i = 0; i < p_keep_count[0]; i++) {
        det = out_nms + i * 7;

        // Output format for each detection is stored in the below order
        // [image_id, label, confidence, xmin, ymin, xmax, ymax]
        if ( det[2] < threshold) continue;
        assert((int) det[1] < out_class_size);

#if 0
        std::cout << "id/label/conf/ x/y x/y -- "
                  << det[0] << " " << det[1] << " " << det[2] << " "
                  << det[3] << " " << det[4] << " " << det[5] << " " << det[6] << std::endl;
#endif
        NvDsInferObjectDetectionInfo object;
            object.classId = (int) det[1];
            object.detectionConfidence = det[2];

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP(det[3] * networkInfo.width, 0, networkInfo.width - 1);
            object.top = CLIP(det[4] * networkInfo.height, 0, networkInfo.height - 1);
            object.width = CLIP((det[5] - det[3]) * networkInfo.width, 0, networkInfo.width - 1);
            object.height = CLIP((det[6] - det[4]) * networkInfo.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomBatchedYoloV5NMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout <<"keep cout"
              <<p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        object.left = CLIP(p_bboxes[4*i], 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1], 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2], 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3], 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomBatchedNMSTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList) {

     if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    /* Host memory for "BatchedNMS"
       BatchedNMS has 4 output bindings, the order is:
       keepCount, bboxes, scores, classes
    */
    int* p_keep_count = (int *) outputLayersInfo[0].buffer;
    float* p_bboxes = (float *) outputLayersInfo[1].buffer;
    float* p_scores = (float *) outputLayersInfo[2].buffer;
    float* p_classes = (float *) outputLayersInfo[3].buffer;

    const float threshold = detectionParams.perClassThreshold[0];

    const int keep_top_k = 200;
    const char* log_enable = std::getenv("ENABLE_DEBUG");

    if(log_enable != NULL && std::stoi(log_enable)) {
        std::cout <<"keep count"
              <<p_keep_count[0] << std::endl;
    }

    for (int i = 0; i < p_keep_count[0] && objectList.size() <= keep_top_k; i++) {

        if ( p_scores[i] < threshold) continue;

        if(log_enable != NULL && std::stoi(log_enable)) {
            std::cout << "label/conf/ x/y x/y -- "
                      << p_classes[i] << " " << p_scores[i] << " "
                      << p_bboxes[4*i] << " " << p_bboxes[4*i+1] << " " << p_bboxes[4*i+2] << " "<< p_bboxes[4*i+3] << " " << std::endl;
        }

        if((unsigned int) p_classes[i] >= detectionParams.numClassesConfigured) continue;
        if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1]) continue;

        NvDsInferObjectDetectionInfo object;
        object.classId = (int) p_classes[i];
        object.detectionConfidence = p_scores[i];

        /* Clip object box co-ordinates to network resolution */
        object.left = CLIP(p_bboxes[4*i] * networkInfo.width, 0, networkInfo.width - 1);
        object.top = CLIP(p_bboxes[4*i+1] * networkInfo.height, 0, networkInfo.height - 1);
        object.width = CLIP(p_bboxes[4*i+2] * networkInfo.width, 0, networkInfo.width - 1) - object.left;
        object.height = CLIP(p_bboxes[4*i+3] * networkInfo.height, 0, networkInfo.height - 1) - object.top;

        if(object.height < 0 || object.width < 0)
            continue;
        objectList.push_back(object);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomMrcnnTLTV2 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList) {
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *detectionLayer = layerFinder("generate_detections");
    const NvDsInferLayerInfo *maskLayer = layerFinder("mask_fcn_logits/BiasAdd");

    if (!detectionLayer || !maskLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }

    if(maskLayer->inferDims.numDims != 4U) {
        std::cerr << "Network output number of dims is : " <<
            maskLayer->inferDims.numDims << " expect is 4"<< std::endl;
        return false;
    }

    const unsigned int det_max_instances = maskLayer->inferDims.d[0];
    const unsigned int num_classes = maskLayer->inferDims.d[1];
    if(num_classes != detectionParams.numClassesConfigured) {
        std::cerr << "WARNING: Num classes mismatch. Configured:" <<
            detectionParams.numClassesConfigured << ", detected by network: " <<
            num_classes << std::endl;
    }
    const unsigned int mask_instance_height= maskLayer->inferDims.d[2];
    const unsigned int mask_instance_width = maskLayer->inferDims.d[3];

    auto out_det = reinterpret_cast<MrcnnRawDetection*>( detectionLayer->buffer);
    auto out_mask = reinterpret_cast<float(*)[mask_instance_width *
        mask_instance_height]>(maskLayer->buffer);

    for(auto i = 0U; i < det_max_instances; i++) {
        MrcnnRawDetection &rawDec = out_det[i];

        if(rawDec.score < detectionParams.perClassPreclusterThreshold[0])
            continue;

        NvDsInferInstanceMaskInfo obj;
        obj.left = CLIP(rawDec.x1, 0, networkInfo.width - 1);
        obj.top = CLIP(rawDec.y1, 0, networkInfo.height - 1);
        obj.width = CLIP(rawDec.x2, 0, networkInfo.width - 1) - rawDec.x1;
        obj.height = CLIP(rawDec.y2, 0, networkInfo.height - 1) - rawDec.y1;
        if(obj.width <= 0 || obj.height <= 0)
            continue;
        obj.classId = static_cast<int>(rawDec.class_id);
        obj.detectionConfidence = rawDec.score;

        obj.mask_size = sizeof(float)*mask_instance_width*mask_instance_height;
        obj.mask = new float[mask_instance_width*mask_instance_height];
        obj.mask_width = mask_instance_width;
        obj.mask_height = mask_instance_height;

        float *rawMask = reinterpret_cast<float *>(out_mask + i
                         * detectionParams.numClassesConfigured + obj.classId);
        memcpy (obj.mask, rawMask, sizeof(float)*mask_instance_width*mask_instance_height);

        objectList.push_back(obj);
    }

    return true;

}

void getMaskDimension(float* buf, int w, int h, int& left, int& top, int& width, int& height)
{
    int right, bottom;
    right = bottom = 0;
    left = top = width = height = 0;
    for(int i = 0; i < w; i++) {
        for(int j = 0; j < h; j++) {
            if(*(buf + j*w + i) == 1) {
                if(left == 0) left = i;
                if(i < left) left = i;
                if(i > right) right = i;
                if(top == 0) top = j;
                if(j < top) top = j;
                if(j > bottom) bottom = j;
            }
            width = right - left;
            height = bottom - top;
        }
    }
}

void copy_mask(float* dst, float* src, int w, int h,
    int mask_left, int mask_top, int mask_width, int mask_height) {
    int j = 0;
    for(int i = mask_top; i < mask_top + mask_height; i++){
        float* pSrc = src + i*w + mask_left;
        memcpy(dst + (j++)*mask_width, pSrc, mask_width*sizeof(float));
    }
}

extern "C"
bool NvDsInferParseCustomMask2Former (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferInstanceMaskInfo> &objectList) {
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if ((layer.dataType == FLOAT || layer.dataType == INT32 || layer.dataType == INT64) &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *pred_classes = layerFinder("pred_classes");
    const NvDsInferLayerInfo *pred_masks = layerFinder("pred_masks");
    const NvDsInferLayerInfo *pred_scores = layerFinder("pred_scores");
    const unsigned int det_max_instances = pred_masks->inferDims.d[0];

    int width = pred_masks->inferDims.d[1];
    int height = pred_masks->inferDims.d[2];
    int* pclass = (int*)pred_classes->buffer;
    float* pmask = (float*)pred_masks->buffer;
    float* pscore = (float*)pred_scores->buffer;
    assert(pclass != NULL && pmask != NULL && pscore != NULL);
    float* tmp_pmask = NULL;
    int mask_left, mask_top, mask_width, mask_height;

    for(auto i = 0U; i < det_max_instances; i++) {
        if(std::isnan(pscore[i]) || pscore[i] < detectionParams.perClassPreclusterThreshold[0])
            continue;
        mask_left = mask_top = mask_width = mask_height = 0;
        tmp_pmask = pmask + i * width * height;
        /* get rect from mask*/
        getMaskDimension(tmp_pmask, width, height, mask_left, mask_top, mask_width, mask_height);
        NvDsInferInstanceMaskInfo obj;
        obj.left = mask_left;
        obj.top = mask_top;
        obj.width = mask_width;
        obj.height = mask_height;
        if(obj.width <= 0 || obj.height <= 0  || mask_width < 0|| mask_height <= 0)
            continue;
        obj.classId = pclass[i];
        obj.detectionConfidence = pscore[i];
        obj.mask_size = sizeof(float)*mask_width*mask_height;
        obj.mask = new float[mask_width*mask_height];
        obj.mask_width = mask_width;
        obj.mask_height = mask_height;
        copy_mask(obj.mask, tmp_pmask, width, height, mask_left, mask_top, mask_width, mask_height);

        objectList.push_back(obj);
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomEfficientDetTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if(outputLayersInfo.size() != 4)
    {
        std::cerr << "Mismatch in the number of output buffers."
                  << "Expected 4 output buffers, detected in the network :"
                  << outputLayersInfo.size() << std::endl;
        return false;
    }

    int* p_keep_count = nullptr;
    float* p_bboxes = nullptr;
    float* p_scores = nullptr;
    int* p_classes = nullptr;

    for (int i = 0; i < 4; i++){
        const char* layerName = outputLayersInfo[i].layerName;
        if(!strcmp(layerName, "num_detections")) {
            p_keep_count = (int *) outputLayersInfo[i].buffer;
        } else if(!strcmp(layerName, "detection_boxes")) {
           p_bboxes = (float *) outputLayersInfo[i].buffer;
        } else if(!strcmp(layerName, "detection_scores")) {
            p_scores = (float *) outputLayersInfo[i].buffer;
        } else if(!strcmp(layerName, "detection_classes")) {
            p_classes = (int *) outputLayersInfo[i].buffer;
        }
    }

    const int out_class_size = detectionParams.numClassesConfigured;
    const float threshold = detectionParams.perClassThreshold[0];

    if (p_keep_count[0] > 0)
    {
        for (int i = 0; i < p_keep_count[0]; i++) {
            if ( p_scores[i] < threshold) continue;
            //assert((int) p_classes[i] < out_class_size);
	    if(p_classes[i] >= out_class_size)
	      break;

            if(p_bboxes[4*i+2] < p_bboxes[4*i] || p_bboxes[4*i+3] < p_bboxes[4*i+1])
                continue;

            NvDsInferObjectDetectionInfo object;
            object.classId = (int) p_classes[i];
            object.detectionConfidence = p_scores[i];

            object.left=p_bboxes[4*i+1];
            object.top=p_bboxes[4*i];
            object.width=( p_bboxes[4*i+3] - object.left);
            object.height= ( p_bboxes[4*i+2] - object.top);

            object.left=CLIP(object.left, 0, networkInfo.width - 1);
            object.top=CLIP(object.top, 0, networkInfo.height - 1);
            object.width=CLIP(object.width, 0, networkInfo.width - 1);
            object.height=CLIP(object.height, 0, networkInfo.height - 1);

            objectList.push_back(object);
        }
    }
    return true;
}

extern "C"
bool NvDsInferParseCustomDDETRTAO (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                NvDsInferNetworkInfo const &networkInfo,
                                NvDsInferParseDetectionParams const &detectionParams,
                                std::vector<NvDsInferObjectDetectionInfo> &objectList) {

    DEBUG_DUMP_SECTION();  // scope to entire function {}

    // Code from NvDsInferParseCustomTfSSD for layer finding
    auto layerFinder = [&outputLayersInfo](const std::string &name)
        -> const NvDsInferLayerInfo *{
        for (auto &layer : outputLayersInfo) {
            if (layer.dataType == FLOAT &&
              (layer.layerName && name == layer.layerName)) {
                return &layer;
            }
        }
        return nullptr;
    };

    const NvDsInferLayerInfo *boxLayer = layerFinder("pred_boxes"); // 1 x num_queries x 4
    const NvDsInferLayerInfo *classLayer = layerFinder("pred_logits");  // 1 x num_queries x num_classes

    if (!boxLayer || !classLayer) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        DEBUG_PRINT("Exiting NvDsInferParseCustomDDETRTAO with error");
        return false;
    }

    // Debug print both tensors
    DEBUG_DS_TENSOR(*boxLayer);
    DEBUG_DS_TENSOR(*classLayer);

    const int keep_top_k = 200;
    unsigned int numDetections = classLayer->inferDims.d[0];
    unsigned int numClasses = classLayer->inferDims.d[1];
    std::map<float, NvDsInferObjectDetectionInfo> ordered_objects;

    size_t numClassesConfigured = detectionParams.perClassPreclusterThreshold.size();
    DEBUG_PRINT("detectionParams.perClassPreclusterThreshold.size(): %ld", numClassesConfigured);
    if(numClassesConfigured != numClasses) {
        std::cerr << "ERROR: numClassesConfigured: " << numClassesConfigured << " != numClasses from output tensor: " << numClasses << std::endl;
        return false;
    }

    for (unsigned int idx = 0; idx < numDetections; idx += 1) {
        NvDsInferObjectDetectionInfo res;

        unsigned int class_layer_idx = idx * numClasses;

        res.classId = std::max_element(((float*)classLayer->buffer+class_layer_idx), ((float*)classLayer->buffer+class_layer_idx+numClasses)) - ((float*)classLayer->buffer+class_layer_idx);
        res.detectionConfidence = ((float*)classLayer->buffer)[class_layer_idx+res.classId];

        // If model does not have sigmoid layer, perform sigmoid calculation here
        res.detectionConfidence = 1.0/(1.0 + exp(-res.detectionConfidence));

        if(res.detectionConfidence < detectionParams.perClassPreclusterThreshold[res.classId]) {
            // use DEBUG_DUMP for intensive logging inside for loop
            DEBUG_DUMP("#%d classId: %d, detectionConfidence: %f, threshold: %f", idx, res.classId, res.detectionConfidence, detectionParams.perClassPreclusterThreshold[res.classId]);
            continue;
        }
        // use DEBUG_DUMP for intensive logging inside for loop
        DEBUG_DUMP("[add] #%d classId: %d, detectionConfidence: %f, threshold: %f", idx, res.classId, res.detectionConfidence, detectionParams.perClassPreclusterThreshold[res.classId]);
        enum {cx, cy, w, h};
        float rectX1f, rectY1f, rectX2f, rectY2f;

        unsigned int box_layer_idx = idx * 4;

        rectX1f = (((float*)boxLayer->buffer)[box_layer_idx + cx] - (((float*)boxLayer->buffer)[box_layer_idx + w]/2)) * networkInfo.width;
        rectY1f = (((float*)boxLayer->buffer)[box_layer_idx + cy] - (((float*)boxLayer->buffer)[box_layer_idx + h]/2)) * networkInfo.height;
        rectX2f = rectX1f + ((float*)boxLayer->buffer)[box_layer_idx + w] * networkInfo.width;
        rectY2f = rectY1f + ((float*)boxLayer->buffer)[box_layer_idx + h] * networkInfo.height;

        rectX1f = CLIP(rectX1f, 0.0f, networkInfo.width - 1);
        rectX2f = CLIP(rectX2f, 0.0f, networkInfo.width - 1);
        rectY1f = CLIP(rectY1f, 0.0f, networkInfo.height - 1);
        rectY2f = CLIP(rectY2f, 0.0f, networkInfo.height - 1);

        res.left = rectX1f;
        res.top = rectY1f;
        res.width = rectX2f - rectX1f;
        res.height = rectY2f - rectY1f;

        ordered_objects[res.detectionConfidence] = res;
    }

    int jdx = 0;
    for (auto iter=ordered_objects.rbegin(); iter!=ordered_objects.rend() && jdx<keep_top_k; iter++, jdx++) {
        if (iter->second.classId != 0){
        objectList.emplace_back(iter->second);}
    }
    DEBUG_PRINT("Found %ld valid detections", objectList.size());
    return true;
}

bool NvDsInferParseCustomSegformerTAO(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {

    DEBUG_DUMP_SECTION();  // scope to entire function {}

    for(auto layer : outputLayersInfo) {
        if(layer.dataType == INT64 && !strcmp(layer.layerName, "output")) {
            DEBUG_PRINT("Processing int64 output layer");
            DEBUG_DS_TENSOR(layer);

            for (unsigned int i = 0; i < networkInfo.height * networkInfo.width; i++) {
                classificationMap[i] = ((int64_t*)layer.buffer)[i];
            }

            DEBUG_TENSOR("segformer_classification_map",
                        classificationMap,
                        networkInfo.height * networkInfo.width,
                        int);
        } else {
            std::cerr << "ERROR: Mismatched data type from output layer: " << layer.layerName << std::endl;
            DEBUG_PRINT("Exiting NvDsInferParseCustomSegformerTAO with error");
            return false;
        }
    }

    return true;
}

extern "C"
bool NvDsInferParseCustomSegformerTAO2(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {

    DEBUG_DUMP_SECTION();  // scope to entire function {}

    if (numClasses <= 0) {
        DEBUG_PRINT("numClasses should be configured greater than 0, but got %d", numClasses);
        return false;
    }

    for(auto layer : outputLayersInfo) {
        if (layer.dataType == FLOAT && !strcmp(layer.layerName, "output")) {
            DEBUG_PRINT("Processing float32 output layer");
            DEBUG_DS_TENSOR(layer);
            DEBUG_PRINT("numClasses=%d, height=%d, width=%d", numClasses, networkInfo.height, networkInfo.width);

            classProbabilityMap = (float*)layer.buffer;
            // Create temporary buffer for argmax result before resize
            unsigned int tensorHeight = layer.inferDims.d[1];  // 56 eg
            unsigned int tensorWidth = layer.inferDims.d[2];   // 56 eg
            std::vector<int> tempClassMap(tensorHeight * tensorWidth);

            // Perform argmax along class dimension
            float maxScore = -1;
            int maxIdx = -1;
            for (unsigned int y = 0; y < tensorHeight; y++) {
                for (unsigned int x = 0; x < tensorWidth; x++) {
                    // find max score and max class index at each pixel
                    for (unsigned int c = 0; c < numClasses; c++) {
                        if (c == 0) {
                            // init maxScore and maxIdx with class 0
                            maxScore = classProbabilityMap[c * tensorWidth * tensorHeight + y * tensorWidth + x];
                            maxIdx = c;
                        } else {
                            float score = classProbabilityMap[c * tensorWidth * tensorHeight + y * tensorWidth + x];
                            if (score > maxScore) {
                                maxScore = score;
                                maxIdx = c;
                            }
                        }
                    }
                    tempClassMap[y * tensorWidth + x] = maxIdx;
                }
            }

            // Nearest neighbor interpolation to resize from tensorHeight x tensorWidth to networkInfo.height x networkInfo.width
            float scaleY = (float)tensorHeight / networkInfo.height;
            float scaleX = (float)tensorWidth / networkInfo.width;

            for (unsigned int y = 0; y < networkInfo.height; y++) {
                for (unsigned int x = 0; x < networkInfo.width; x++) {
                    // Find nearest source pixel
                    unsigned int srcY = std::min((unsigned int)(y * scaleY), tensorHeight - 1);
                    unsigned int srcX = std::min((unsigned int)(x * scaleX), tensorWidth - 1);

                    // Direct copy of class index
                    classificationMap[y * networkInfo.width + x] = tempClassMap[srcY * tensorWidth + srcX];
                }
            }
            DEBUG_TENSOR("segformer_classification_map",
                        classificationMap,
                        networkInfo.height * networkInfo.width,
                        int);
        } else {
            std::cerr << "ERROR: Mismatched data type from output layer: " << layer.layerName << std::endl;
            DEBUG_PRINT("Exiting NvDsInferParseCustomSegformerTAO2 with error");
            return false;
        }
    }

    return true;
}

bool NvDsInferParseCustomChangeNet(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    float segmentationThreshold,
    unsigned int numClasses,
    int* classificationMap,
    float*& classProbabilityMap) {
    bool found = false;
    for(auto layer : outputLayersInfo) {
        if(layer.dataType == FLOAT && !strcmp(layer.layerName, "output_final")) {
            classProbabilityMap = (float*)layer.buffer;
            for (unsigned int y = 0; y < networkInfo.height; y++) {
                for (unsigned int x = 0; x < networkInfo.width; x++) {
                    float max_prob = -1;
                    int& cls = classificationMap[y * networkInfo.width + x] = -1;
                    for (unsigned int c = 0; c < numClasses; c++) {
                        float prob = classProbabilityMap[c * networkInfo.width * networkInfo.height + y * networkInfo.width + x];
                        if (prob > max_prob && prob > segmentationThreshold) {
                            cls = c;
                            max_prob = prob;
                        }
                    }
                }
            }
            found = true;
            break;
        }
    }
    if (!found) {
        std::cerr << "ERROR: output_final layer not found" << std::endl;
        return false;
    }
    return true;
}

extern "C"
bool NvDsInferClassiferParseNonSoftmax (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &attrString) {
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = outputLayersInfo.size();
    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++)
    {
        const NvDsInferLayerInfo &layerInfo = outputLayersInfo[l];
        /* Check if the layer is of type FLOAT. */
        if (layerInfo.dataType != FLOAT)
        {
            std::cerr << "ERROR: Mismatched data type from output layer: " << layerInfo.layerName << std::endl;
            return false;
        }
        unsigned int numClasses = layerInfo.inferDims.d[0];
        float *outputCoverageBuffer = (float *)layerInfo.buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;
        float sum = 0;

        for (unsigned int c = 0; c < numClasses; c++) {
            sum += exp(outputCoverageBuffer[c]);
        }
        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++)
        {
            // Calculate the softmax probability
            float probability = exp(outputCoverageBuffer[c]) / sum;
            if (probability > classifierThreshold
                    && probability > maxProbability)
            {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
                attr.attributeLabel = nullptr;
            }
        }
        if (attrFound) {
            attrList.push_back(attr);
        }
    }
    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomNMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBatchedYoloV5NMSTLT);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomBatchedNMSTLT);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMrcnnTLTV2);
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMask2Former);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomEfficientDetTAO);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDDETRTAO);
CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSegformerTAO);
CHECK_CUSTOM_SEM_SEGMENTATION_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSegformerTAO2);
CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassiferParseNonSoftmax);
