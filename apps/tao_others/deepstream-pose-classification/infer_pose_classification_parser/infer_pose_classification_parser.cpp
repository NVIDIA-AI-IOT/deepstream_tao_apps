/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <locale>
#include <codecvt>
#include "nvdsinfer.h"
#include <fstream>
#include <cmath>
using namespace std;
using std::string;
using std::vector;

const char classes_str[6][32] = {
  "sitting_down", "getting_up", "sitting", "standing", "walking", "jumping"
};

extern "C"
{

/* convert poseclassifiction's ouptut logits to probalilities */
std::vector<float>
logits_to_probabilities(float *array, int arrayLen)
{
    std::vector<float> ret;
    float sum = 0;
    for (int i = 0; i < arrayLen; i++) {
        sum += exp(array[i]);
    }
    for (int i = 0; i < arrayLen; i++) {
        float probability = exp(array[i]);
        probability /= sum;
        ret.push_back(probability);
    }
    return ret;
}

bool NvDsParseCustomPoseClassification(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                 NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                 std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
{
    float maxProbability = 0;
    bool attrFound = false;
    NvDsInferAttribute attr;

    NvDsInferDimsCHW dims;
    getDimsCHWFromDims (dims, outputLayersInfo[0].inferDims);
    unsigned int numClasses = dims.c;
    float *outputCoverageBuffer = (float *) outputLayersInfo[0].buffer;
    /* Iterate through all the probabilities that the object belongs to
        * each class. Find the maximum probability and the corresponding class
        * which meets the minimum threshold. */
    std::vector<float> probabilities;
    probabilities = logits_to_probabilities(outputCoverageBuffer, numClasses);
    for (unsigned int c = 0; c < numClasses; c++) {
        float probability = probabilities[c];
        if (probability > classifierThreshold && probability > maxProbability) {
            //printf("c:%d, probability:%f\n", c, probability);
            maxProbability = probability;
            attrFound = true;
            attr.attributeIndex = 0;
            attr.attributeValue = c;
            attr.attributeConfidence = probability;
            attrString = classes_str[attr.attributeValue];
        }
    }

    if (attrFound) {
        attr.attributeLabel = strdup(attrString.c_str());
        attrList.push_back(attr);
    }

    return true;
}

}//end of extern "C"
