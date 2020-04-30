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
#include "nvdsinfer_custombboxparser_frcnn_tlt.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

static params gParams;

/* This is a sample bounding box parsing function for the sample faster RCNN
 *
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomFrcnnTLT (
         std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
         NvDsInferNetworkInfo  const &networkInfo,
         NvDsInferParseDetectionParams const &detectionParams,
         std::vector<NvDsInferObjectDetectionInfo> &objectList);

std::vector<int> nms_classifier(std::vector<float>& boxes_per_cls,
                                std::vector<float>& probs_per_cls,
                                float NMS_OVERLAP_THRESHOLD,
                                int NMS_MAX_BOXES) {
    int num_boxes = boxes_per_cls.size() / 4;
    std::vector<std::pair<float, int>> score_index;

    for (int i = 0; i < num_boxes; ++i) {
        score_index.push_back(std::make_pair(probs_per_cls[i], i));
    }

    std::stable_sort(score_index.begin(), score_index.end(),
    [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {
        return pair1.first > pair2.first;
    });

    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }

        return x1max < x2min ? 0.0f : (std::min(x1max, x2max) - x2min + 1.0f);
    };

    auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0] + 1.0f) * (bbox1[3] - bbox1[1] + 1.0f);
        float area2 = (bbox2[2] - bbox2[0] + 1.0f) * (bbox2[3] - bbox2[1] + 1.0f);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };
    std::vector<int> indices;

    for (auto i : score_index) {
        const int idx = i.second;
        bool keep = true;

        for (unsigned k = 0; k < indices.size(); ++k) {
            if (keep) {
                const int kept_idx = indices[k];
                float overlap = computeIoU(&boxes_per_cls[idx * 4], &boxes_per_cls[kept_idx * 4]);
                keep = overlap <= NMS_OVERLAP_THRESHOLD;
            } else {
                break;
            }
        }

        if (indices.size() >= static_cast<unsigned>(NMS_MAX_BOXES)) {
            break;
        }

        if (keep) {
            indices.push_back(idx);
        }
    };

    return indices;
}

void batch_inverse_transform_classifier(
         const float* roi_after_nms, int roi_num_per_img,
         const float* classifier_cls, const float* classifier_regr,
         std::vector<float>& pred_boxes, std::vector<int>& pred_cls_ids,
         std::vector<float>& pred_probs, std::vector<int>& box_num_per_img,
         int N) {
    auto max_index = [](const float* start, const float* end) -> int {
        float max_val = start[0];
        int max_pos = 0;

        for (int i = 1; start + i < end; ++i) {
            if (start[i] > max_val) {
                max_val = start[i];
                max_pos = i;
            }
        }

        return max_pos;
    };
    int box_num;

    for (int n = 0; n < N; ++n) {
        box_num = 0;

        for (int i = 0; i < roi_num_per_img; ++i) {
            auto max_idx = max_index(
                     classifier_cls + n * roi_num_per_img * gParams.outputClassSize + i * gParams.outputClassSize,
                     classifier_cls + n * roi_num_per_img * gParams.outputClassSize + i * gParams.outputClassSize +
                     gParams.outputClassSize);

            if (max_idx == (gParams.outputClassSize - 1) ||
                classifier_cls[n * roi_num_per_img * gParams.outputClassSize + max_idx + i * gParams.outputClassSize] <
                gParams.visualizeThreshold) {
                continue;
            }

            // Inverse transform
            float tx, ty, tw, th;
            //(i, 20, 4)
            tx = classifier_regr[n * roi_num_per_img * gParams.outputBboxSize + i * gParams.outputBboxSize + max_idx * 4]
                     / gParams.classifierRegressorStd[0];
            ty = classifier_regr[n * roi_num_per_img * gParams.outputBboxSize + i * gParams.outputBboxSize + max_idx * 4 + 1]
                     / gParams.classifierRegressorStd[1];
            tw = classifier_regr[n * roi_num_per_img * gParams.outputBboxSize + i * gParams.outputBboxSize + max_idx * 4 + 2]
                     / gParams.classifierRegressorStd[2];
            th = classifier_regr[n * roi_num_per_img * gParams.outputBboxSize + i * gParams.outputBboxSize + max_idx * 4 + 3]
                     / gParams.classifierRegressorStd[3];
            float y = roi_after_nms[n * roi_num_per_img * 4 + 4 * i] * static_cast<float>(gParams.inputHeight - 1.0f);
            float x = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 1] * static_cast<float>(gParams.inputWidth - 1.0f);
            float ymax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 2] * static_cast<float>(gParams.inputHeight - 1.0f);
            float xmax = roi_after_nms[n * roi_num_per_img * 4 + 4 * i + 3] * static_cast<float>(gParams.inputWidth - 1.0f);
            float w = xmax - x;
            float h = ymax - y;
            float cx = x + w / 2.0f;
            float cy = y + h / 2.0f;
            float cx1 = tx * w + cx;
            float cy1 = ty * h + cy;
            float w1 = std::exp(static_cast<double>(tw)) * w;
            float h1 = std::exp(static_cast<double>(th)) * h;
            float x1 = cx1 - w1 / 2.0f;
            float y1 = cy1 - h1 / 2.0f;
            auto clip
                = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
            float x2 = x1 + w1;
            float y2 = y1 + h1;
            x1 = clip(x1, 0.0f, gParams.inputWidth - 1.0f);
            y1 = clip(y1, 0.0f, gParams.inputHeight - 1.0f);
            x2 = clip(x2, 0.0f, gParams.inputWidth - 1.0f);
            y2 = clip(y2, 0.0f, gParams.inputHeight - 1.0f);

            if (x2 > x1 && y2 > y1) {
                pred_boxes.push_back(x1);
                pred_boxes.push_back(y1);
                pred_boxes.push_back(x2);
                pred_boxes.push_back(y2);
                pred_probs.push_back(classifier_cls[n * roi_num_per_img * gParams.outputClassSize +
                                                    max_idx + i * gParams.outputClassSize]);
                pred_cls_ids.push_back(max_idx);
                ++box_num;
            }
        }

        box_num_per_img.push_back(box_num);
    }
}

void parse_boxes(int img_num, int class_num,
         std::vector<float>& pred_boxes, std::vector<float>& pred_probs,
         std::vector<int>& pred_cls_ids, std::vector<int>& box_num_per_img,
         std::vector<std::vector<Detection>>& results) {
    int box_start_idx = 0;
    std::vector<float> boxes_per_cls;
    std::vector<float> probs_per_cls;
    std::vector<Detection> det_per_img;

    for (int i = 0; i < img_num; ++i) {
        det_per_img.clear();

        for (int c = 0; c < (class_num - 1); ++c) {
            // skip the background
            boxes_per_cls.clear();
            probs_per_cls.clear();

            for (int k = box_start_idx; k < box_start_idx + box_num_per_img[i]; ++k) {
                if (pred_cls_ids[k] == c) {
                    boxes_per_cls.push_back(pred_boxes[4 * k]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 1]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 2]);
                    boxes_per_cls.push_back(pred_boxes[4 * k + 3]);
                    probs_per_cls.push_back(pred_probs[k]);
                }
            }

            // Apply NMS algorithm per class
            auto indices_after_nms
                = nms_classifier(boxes_per_cls, probs_per_cls, 
                      gParams.nmsIouThresholdClassifier, gParams.postNmsTopN);

            // Show results
            for (unsigned k = 0; k < indices_after_nms.size(); ++k) {
                int idx = indices_after_nms[k];
                Detection b{boxes_per_cls[idx * 4], boxes_per_cls[idx * 4 + 1], boxes_per_cls[idx * 4 + 2],
                            boxes_per_cls[idx * 4 + 3], c, probs_per_cls[idx]};
                det_per_img.push_back(b);
            }
        }

        box_start_idx += box_num_per_img[i];
        results.push_back(det_per_img);
    }
}

bool ParseOutput(const int batchSize,
                 const float* out_class,
                 const float* out_reg,
                 const float* out_proposal,
                 std::vector<std::vector<Detection>>& results) {
    const int outputClassSize = gParams.outputClassSize;
    std::vector<float> classifierRegressorStd;
    std::vector<std::string> classNames;
    std::vector<float> pred_boxes;
    std::vector<int> pred_cls_ids;
    std::vector<float> pred_probs;
    std::vector<int> box_num_per_img;
    results.clear();

    int post_nms_top_n = gParams.postNmsTopN;
    
    // Post processing for stage 2.
    batch_inverse_transform_classifier(out_proposal, post_nms_top_n, out_class, out_reg, pred_boxes, pred_cls_ids,
                                       pred_probs, box_num_per_img, batchSize);
    parse_boxes(batchSize, outputClassSize, pred_boxes, pred_probs, pred_cls_ids, box_num_per_img, results);
    return true;
}


extern "C"
bool NvDsInferParseCustomFrcnnTLT (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo  const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    static NvDsInferDimsCHW covLayerDims;

    static int proposalIndex = -1;
    static int bboxLayerIndex = -1;
    static int covLayerIndex = -1;

    static bool classMismatchWarn = false;

    /* Find the proposal layer */
    if (proposalIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "proposal") == 0) {
                proposalIndex = i;
                break;
            }
        }
        if (proposalIndex == -1) {
            std::cerr << "Could not find proposal layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the bbox layer */
    if (bboxLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "dense_regress_td/BiasAdd") == 0) {
                bboxLayerIndex = i;
                break;
            }
        }
        if (bboxLayerIndex == -1) {
            std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Find the cov layer */
    if (covLayerIndex == -1) {
        for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
            if (strcmp(outputLayersInfo[i].layerName, "dense_class_td/Softmax") == 0) {
                covLayerIndex = i;
                getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].inferDims);
                break;
            }
        }
        if (covLayerIndex == -1) {
            std::cerr << "Could not find cov layer buffer while parsing" << std::endl;
            return false;
        }
    }

    /* Warn in case of mismatch in number of classes */
    if (!classMismatchWarn) {
        if (covLayerDims.h != detectionParams.numClassesConfigured) {
            std::cerr << "WARNING: Num classes mismatch. Configured:" <<
                      detectionParams.numClassesConfigured << ", detected by network: " <<
                      covLayerDims.c << " " << covLayerDims.h << " " << covLayerDims.w << std::endl;
        }
        classMismatchWarn = true;
    }

    gParams.inputChannels = networkInfo.channels;
    gParams.inputHeight = networkInfo.height;
    gParams.inputWidth = networkInfo.width;
    gParams.nmsIouThresholdClassifier = 0.3f;
    gParams.visualizeThreshold = detectionParams.perClassThreshold[0];
    gParams.classifierRegressorStd.push_back(10.0f);
    gParams.classifierRegressorStd.push_back(10.0f);
    gParams.classifierRegressorStd.push_back(5.0f);
    gParams.classifierRegressorStd.push_back(5.0f);
    gParams.outputClassSize = detectionParams.numClassesConfigured;
    gParams.outputBboxSize = (gParams.outputClassSize - 1) * 4;
    gParams.postNmsTopN = 300;

    // Host memory for "proposal"
    const float* out_proposal = (float *) outputLayersInfo[proposalIndex].buffer;
    
    // Host memory for "dense_class_4/Softmax"
    const float* out_class = (float *) outputLayersInfo[covLayerIndex].buffer;
    
    // Host memory for "dense_regress_4/BiasAdd"
    const float* out_reg = (float *) outputLayersInfo[bboxLayerIndex].buffer;

    const int batch_size = 1;

    std::vector<std::vector<Detection> > results;
    ParseOutput(batch_size, out_class, out_reg, out_proposal, results);

    for (unsigned int i = 0; i < results.size(); i++) {
        for (unsigned int j = 0; j < results[i].size(); j++) {
            NvDsInferObjectDetectionInfo object;
            object.classId = results[i][j].cls;
            object.detectionConfidence = results[i][j].conf;

            /* Clip object box co-ordinates to network resolution */
            object.left = CLIP(results[i][j].x1, 0, networkInfo.width - 1);
            object.top = CLIP(results[i][j].y1, 0, networkInfo.height - 1);
            object.width = CLIP(results[i][j].x2 - results[i][j].x1, 0, networkInfo.width - 1);
            object.height = CLIP(results[i][j].y2 - results[i][j].y1, 0, networkInfo.height - 1);

            objectList.push_back(object);
        }
    }

    return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomFrcnnTLT);
