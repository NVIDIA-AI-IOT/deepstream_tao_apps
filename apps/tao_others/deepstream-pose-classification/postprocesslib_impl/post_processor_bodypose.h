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

#ifndef __POST_PROCESSOR_BODYPOSE_HPP__
#define __POST_PROCESSOR_BODYPOSE_HPP__

#include "post_processor.h"
#include "nvdsmeta_schema.h"

#include <cmath>

class OneEuroFilter {
public:
  /// Default constructor
  OneEuroFilter() {
    reset(30.0f /* Hz */, 0.1f /* Hz */, 0.09f /* ??? */, 0.5f /* Hz */);
  }
  /// Constructor
  /// @param dataUpdateRate   the sampling rate, i.e. the number of samples per unit of time.
  /// @param minCutoffFreq    the lowest bandwidth filter applied.
  /// @param cutoffSlope      the rate at which the filter adapts: higher levels reduce lag.
  /// @param derivCutoffFreq  the bandwidth of the filter applied to smooth the derivative, default 1 Hz.
  OneEuroFilter(float dataUpdateRate, float minCutoffFreq, float cutoffSlope, float derivCutoffFreq) {
    reset(dataUpdateRate, minCutoffFreq, cutoffSlope, derivCutoffFreq);
  }
  /// Reset all parameters of the filter.
  /// @param dataUpdateRate   the sampling rate, i.e. the number of samples per unit of time.
  /// @param minCutoffFreq    the lowest bandwidth filter applied.
  /// @param cutoffSlope      the rate at which the filter adapts: higher levels reduce lag.
  /// @param derivCutoffFreq  the bandwidth of the filter applied to smooth the derivative, default 1 Hz.
  void reset(float dataUpdateRate, float minCutoffFreq, float cutoffSlope, float derivCutoffFreq) {
    reset(); _rate = dataUpdateRate; _minCutoff = minCutoffFreq; _beta = cutoffSlope; _dCutoff = derivCutoffFreq;
  }
  /// Reset only the initial condition of the filter, leaving parameters the same.
  void reset() { _firstTime = true; _xFilt.reset(); _dxFilt.reset(); }
  /// Apply the one euro filter to the given input.
  /// @param x  the unfiltered input value.
  /// @return   the filtered output value.
  float filter(float x)
  {
    float dx, edx, cutoff;
    if (_firstTime) {
      _firstTime = false;
      dx = 0;
    } else {
      dx = (x - _xFilt.hatXPrev()) * _rate;
    }
    edx = _dxFilt.filter(dx, alpha(_rate, _dCutoff));
    cutoff = _minCutoff + _beta * fabsf(edx);
    return _xFilt.filter(x, alpha(_rate, cutoff));
  }


private:
  class LowPassFilter {
  public:
    LowPassFilter() { reset(); }
    void reset() { _firstTime = true; }
    float hatXPrev() const { return _hatXPrev; }
    float filter(float x, float alpha){
      if (_firstTime) {
        _firstTime = false;
        _hatXPrev = x;
      }
      float hatX = alpha * x + (1.f - alpha) * _hatXPrev;
      _hatXPrev = hatX;
      return hatX;

    }
  private:
    float _hatXPrev;
    bool _firstTime;
  };
  inline float alpha(float rate, float cutoff) {
  const float kOneOverTwoPi = 0.15915494309189533577f;  // 1 / (2 * pi)
  // The paper has 4 divisions, but we only use one
  // float tau = kOneOverTwoPi / cutoff, te = 1.f / rate;
  // return 1.f / (1.f + tau / te);
  return cutoff / (rate * kOneOverTwoPi + cutoff);
}
  bool _firstTime;
  float _rate, _minCutoff, _dCutoff, _beta;
  LowPassFilter _xFilt, _dxFilt;
};

constexpr int NVDS_OBJECT_TYPE_PERSON_EXT_POSE  = 0x103;

typedef struct NvDsPersonPoseExt {
  gint num_poses;
  NvDsJoints *poses;
}NvDsPersonPoseExt;


class BodyPoseModelPostProcessor : public ModelPostProcessor{

public:
  BodyPoseModelPostProcessor(int id, int gpuId = 0)
    : ModelPostProcessor (NvDsPostProcessNetworkType_BodyPose, id, gpuId) {}

  ~BodyPoseModelPostProcessor() override = default;

  NvDsPostProcessStatus
  initResource(NvDsPostProcessContextInitParams& initParams) override;

  NvDsPostProcessStatus parseEachFrame(const std::vector <NvDsInferLayerInfo>&
       outputLayers,
       NvDsPostProcessFrameOutput &result) override;

  void
    attachMetadata     (NvBufSurface *surf, gint batch_idx,
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
    gboolean symmetric_padding)
    override;

  void
    prcoessMetadata (NvDsInferTensorMeta *tensor_meta,
      NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta) override;

  void releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput) override;
private:
  NvDsPostProcessStatus fillBodyPoseOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output);
  void osdBody(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[]);

  void movenetposeFromTensorMeta(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output);

  float median(std::vector<float>& v);

  float m_ClassificationThreshold = 0.50;
  std::unordered_map<gint , std::vector<OneEuroFilter>> m_filter_pose;
  OneEuroFilter m_filterRootDepth; // Root node in pose25d.
  fpos_t m_fp_25_pos;
};

#endif
