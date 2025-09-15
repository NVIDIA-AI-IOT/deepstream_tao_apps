/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __POST_PROCESSOR_HPP__
#define __POST_PROCESSOR_HPP__

#include <iostream>
#include <fstream>
#include <thread>
#include <cstring>
#include <queue>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <condition_variable>
#include <yaml-cpp/yaml.h>
#include <limits.h>
#include <cassert>
#include <algorithm>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"
#include "nvdsinfer_dbscan.h"
#include "post_processor_struct.h"


#ifndef PP_DISABLE_CLASS_COPY
#define PP_DISABLE_CLASS_COPY(NoCopyClass)  \
    NoCopyClass(const NoCopyClass&) = delete; \
    void operator=(const NoCopyClass&) = delete
#endif


class ModelPostProcessor
{

protected:
  ModelPostProcessor(NvDsPostProcessNetworkType type, int id, int gpuId)
    : m_NetworkType(type), m_UniqueID(id), m_GpuID(gpuId){}

public:
  virtual ~ModelPostProcessor() = default;

  virtual NvDsPostProcessStatus
  initResource(NvDsPostProcessContextInitParams& initParams);
  const std::vector<std::vector<std::string>>& getLabels() const
  {
    return m_Labels;
  }
  void freeBatchOutput(NvDsPostProcessBatchOutput& batchOutput);
  void setNetworkInfo (NvDsInferNetworkInfo networkInfo){
    m_NetworkInfo = networkInfo;
  }

  virtual NvDsPostProcessStatus parseEachFrame(
      const std::vector <NvDsInferLayerInfo> &outputLayers,
      NvDsPostProcessFrameOutput& result) = 0;


  virtual void
    attachMetadata (NvBufSurface *surf, gint batch_idx,
    NvDsBatchMeta  *batch_meta,
    NvDsFrameMeta  *frame_meta,
    NvDsObjectMeta  *object_meta,
    NvDsObjectMeta *parent_obj_meta,
    NvDsPostProcessFrameOutput & detection_output,
    NvDsPostProcessDetectionParams *all_params,
    std::set <gint> & filterOutClassIds,
    int32_t unique_id,
    gboolean output_instance_mask,
    gboolean process_full_frame,
    float segmentationThreshold,
    gboolean maintain_aspect_ratio,
    NvDsRoiMeta *roi_meta,
    gboolean symmetric_padding) = 0;

  virtual void
    prcoessMetadata (NvDsInferTensorMeta *tensor_meta,
      NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta) = 0;

  virtual void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) = 0;

protected:
  NvDsPostProcessStatus parseLabelsFile(const std::string &path);

private:
  PP_DISABLE_CLASS_COPY(ModelPostProcessor);

protected:
  /* Processor type */
  NvDsPostProcessNetworkType m_NetworkType = NvDsPostProcessNetworkType_Other;

  int m_UniqueID = 0;
  uint32_t m_GpuID = 0;

  /* Network input information. */
  NvDsInferNetworkInfo m_NetworkInfo = {0};
  std::vector<NvDsInferLayerInfo> m_AllLayerInfo;
  std::vector<NvDsInferLayerInfo> m_OutputLayerInfo;

  /* Holds the string labels for classes. */
  std::vector<std::vector<std::string>> m_Labels;

};

#endif
