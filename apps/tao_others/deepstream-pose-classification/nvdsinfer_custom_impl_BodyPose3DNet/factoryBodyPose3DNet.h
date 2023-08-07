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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "nvdssample_BodyPose3DNet_common.h"

#include <cassert>
#include <cstring>
#include <memory>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

class FRCNNPluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:
  virtual nvinfer1::IPluginV2* createPlugin(const char* layerName,
      const nvinfer1::Weights* weights, int nbWeights,
      const char* libNamespace) noexcept override
  {
    assert(isPluginV2(layerName));
    if (!strcmp(layerName, "RPROIFused"))
    {
      assert(mPluginRPROI == nullptr);
      assert(nbWeights == 0 && weights == nullptr);

      auto creator = getPluginRegistry()->getPluginCreator("RPROI_TRT", "1");

      nvinfer1::PluginField fields[]{
          {"poolingH", &poolingH,  nvinfer1::PluginFieldType::kINT32, 1},
          {"poolingW", &poolingW,  nvinfer1::PluginFieldType::kINT32, 1},
          {"featureStride", &featureStride,  nvinfer1::PluginFieldType::kINT32, 1},
          {"preNmsTop", &preNmsTop,  nvinfer1::PluginFieldType::kINT32, 1},
          {"nmsMaxOut", &nmsMaxOut,  nvinfer1::PluginFieldType::kINT32, 1},
          {"iouThreshold", &iouThreshold,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"minBoxSize", &minBoxSize,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"spatialScale", &spatialScale,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"anchorsRatioCount", &anchorsRatioCount,  nvinfer1::PluginFieldType::kINT32, 1},
          {"anchorsRatios", anchorsRatios,  nvinfer1::PluginFieldType::kFLOAT32, 1},
          {"anchorsScaleCount", &anchorsScaleCount,  nvinfer1::PluginFieldType::kINT32, 1},
          {"anchorsScales", anchorsScales,  nvinfer1::PluginFieldType::kFLOAT32, 1},
      };
      nvinfer1::PluginFieldCollection pluginData;
      pluginData.nbFields = 12;
      pluginData.fields = fields;

      mPluginRPROI = std::unique_ptr<IPluginV2, decltype(pluginDeleter)>(
              creator->createPlugin(layerName, &pluginData),
              pluginDeleter);
      mPluginRPROI.get()->setPluginNamespace(libNamespace);
      return mPluginRPROI.get();
    }
    else
    {
      assert(0);
      return nullptr;
    }
  }

  // caffe parser plugin implementation
  bool isPluginV2(const char* name) noexcept override { return !strcmp(name, "RPROIFused"); }

  void destroyPlugin()
  {
    mPluginRPROI.reset();
  }

  void (*pluginDeleter)(IPluginV2*){[](IPluginV2* ptr) { ptr->destroy(); }};
  std::unique_ptr<IPluginV2, decltype(pluginDeleter)> mPluginRPROI{nullptr, pluginDeleter};

  virtual ~FRCNNPluginFactory()
  {
  }
};
