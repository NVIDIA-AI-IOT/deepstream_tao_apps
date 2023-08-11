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
#include "factoryBodyPose3DNet.h"

bool NvDsInferPluginFactoryCaffeGet (NvDsInferPluginFactoryCaffe &pluginFactory,
    NvDsInferPluginFactoryType &type)
{
  type = PLUGIN_FACTORY_V2;
  pluginFactory.pluginFactoryV2 = new FRCNNPluginFactory;

  return true;
}

void NvDsInferPluginFactoryCaffeDestroy (NvDsInferPluginFactoryCaffe &pluginFactory)
{
  FRCNNPluginFactory *factory =
      static_cast<FRCNNPluginFactory *> (pluginFactory.pluginFactoryV2);
  factory->destroyPlugin();
  delete factory;
}
