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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <iostream>

struct Detection {
    float x1, y1, x2, y2;
    int cls ;
    float conf;
};

struct params {
    std::string outputClsName;
    std::string outputRegName;
    std::string outputProposalName;
    std::string inputNodeName;

    int inputChannels;
    int inputHeight;
    int inputWidth;
    int outputClassSize;
    float nmsIouThresholdClassifier;
    float visualizeThreshold;
    int postNmsTopN;
    int outputBboxSize;
    std::vector<float> classifierRegressorStd;

};

std::vector<int> nms_classifier(std::vector<float>& boxes_per_cls,
                                std::vector<float>& probs_per_cls, float NMS_OVERLAP_THRESHOLD, int NMS_MAX_BOXES);

void batch_inverse_transform_classifier(const float* roi_after_nms, int roi_num_per_img,
                                        const float* classifier_cls, const float* classifier_regr, std::vector<float>& pred_boxes,
                                        std::vector<int>& pred_cls_ids, std::vector<float>& pred_probs, std::vector<int>& box_num_per_img, int N);

void parse_boxes(int img_num, int class_num, std::vector<float>& pred_boxes,
                 std::vector<float>& pred_probs, std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
                 std::vector<std::vector<Detection>>& results);

bool ParseOutput(const int batchSize,
                 const float* out_class,
                 const float* out_reg,
                 const float* out_proposal,
                 std::vector<std::vector<Detection>>& results);

