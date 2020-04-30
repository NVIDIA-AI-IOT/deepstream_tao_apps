/**
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"
#include <cassert>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

//static params gParams;

/* This is a sample bounding box parsing function for the sample ssd
 *
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomSSDTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

extern "C"
bool NvDsInferParseCustomSSDTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    static int nmsIndex = -1;

    /* Find the nms layer */
    if (nmsIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "NMS") == 0) {
                nmsIndex = i;
                break;
            }
        }
        if (nmsIndex == -1) {
            std::cerr << "Could not find nms layer buffer while parsing" << std::endl;
            return false;
        }
    }

    // Host memory for "nms"
    float* out_nms = (float *) outputLayersInfo[nmsIndex].buffer;
 
    const int batch_id = 0;
    const int out_class_size = detectionParams.numClassesConfigured;
    const float threshold = detectionParams.perClassThreshold[0];

    // Set your keep_count / keep_top here
    const int keep_count = 200;
    const int keep_top_k = 200;

    float* det;

    for (int i = 0; i < keep_count; i++) {
        det = out_nms + batch_id * keep_top_k * 7 + i * 7;

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

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomSSDTLT);
