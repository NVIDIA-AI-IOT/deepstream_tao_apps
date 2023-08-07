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

#include "nvdsinfer_custom_impl.h"
#include <cstring>
/* Assumes only one input layer "im_info" needs to be initialized */
bool NvDsInferInitializeInputLayers (std::vector<NvDsInferLayerInfo> const &inputLayersInfo,
        NvDsInferNetworkInfo const &networkInfo,
        unsigned int maxBatchSize)
{
  float scale_normalized_mean_limb_lengths[] = {
     0.5000, 0.5000, 1.0000, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3433, 0.8171,
     0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.3422, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
 float mean_limb_lengths[] =  {
     246.3427, 246.3427, 492.6854, 402.4380, 487.0321, 128.6856, 391.6295,
     281.9928, 249.9478,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
     0.0000,   0.0000, 169.1832, 402.2611, 488.1824, 128.6848, 407.5836,
     281.9897, 249.9489,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
     0.0000,   0.0000, 168.6137,   0.0000,   0.0000,   0.0000,   0.0000,
     0.0000};

  //k_inv would change for camera parameters
  float k_inv[] = {0.00124876620338,   0,                -0.119881555525,
                   0,                  0.00124876620338, -0.159842074033,
                   0,                  0,                 1};

  float t_form_inv[] = {1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0};
  for (auto v : inputLayersInfo){
    if (!strcmp(v.layerName, "scale_normalized_mean_limb_lengths")){
      memcpy(v.buffer,scale_normalized_mean_limb_lengths,sizeof(float)*36);
    }
    if (!strcmp(v.layerName, "mean_limb_lengths")){
      memcpy(v.buffer,mean_limb_lengths,sizeof(float)*36);
    }
    if (!strcmp(v.layerName, "k_inv")){
      memcpy(v.buffer,k_inv,sizeof(float)*9);
    }
    if (!strcmp(v.layerName, "t_form_inv")){
      memcpy(v.buffer,t_form_inv,sizeof(float)*9);
    }
  }

  return true;
}

