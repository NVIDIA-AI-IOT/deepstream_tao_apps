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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <signal.h>
#include <bits/stdc++.h>

#include "cuda_runtime_api.h"
#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvdsgstutils.h"
#include "nvbufsurface.h"
#include "nvdsmeta_schema.h"
#include "deepstream_common.h"
#include "deepstream_perf.h"
#include "nvds_yml_parser.h"
#include <yaml-cpp/yaml.h>
#include <sys/time.h>
#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <ctime>

#define EPS 1e-6
#define MAX_DISPLAY_LEN 64

// Default camera attributes
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720
#define FOCAL_LENGTH 800.79041f
#define _PATH_MAX 1024

/* Padding due to AR SDK model requires bigger bboxes*/
#define PAD_DIM 128

#define PGIE_CLASS_ID_PERSON 0

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

//---Global variables derived from program arguments---
static guint _cintr = FALSE;
static gboolean _quit = FALSE;
double _focal_length_dbl = FOCAL_LENGTH;
float _focal_length = (float)_focal_length_dbl;
int _image_width = MUXER_OUTPUT_WIDTH;
int _image_height = MUXER_OUTPUT_HEIGHT;
int _pad_dim = PAD_DIM;// A scaled version of PAD_DIM
Eigen::Matrix3f _K;// Camera intrinsic matrix
//---Global variables derived from program arguments---

static GstElement *pipeline = NULL;

gint frame_number = 0;

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

//===Global variables===
std::unordered_map<gint , std::vector<OneEuroFilter>> g_filter_pose25d;
OneEuroFilter m_filterRootDepth; // Root node in pose25d.

fpos_t g_fp_25_pos;

//===Global variables===

#define ACQUIRE_DISP_META(dmeta)  \
  if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META  || \
      dmeta->num_labels == MAX_ELEMENTS_IN_DISPLAY_META ||  \
      dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) \
        { \
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);\
          nvds_add_display_meta_to_frame(frame_meta, dmeta);\
        }\

static float _sgie_classifier_threshold = FLT_MIN;

typedef struct NvAR_Point3f {
  float x, y, z;
} NvAR_Point3f;

static Eigen::Matrix3f m_K_inv_transpose;
const float m_scale_ll[] = {
  0.5000, 0.5000, 1.0000, 0.8175, 0.9889, 0.2610, 0.7942, 0.5724, 0.5078,
  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3433, 0.8171,
  0.9912, 0.2610, 0.8259, 0.5724, 0.5078, 0.0000, 0.0000, 0.0000, 0.0000,
  0.0000, 0.0000, 0.0000, 0.3422, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
const float m_mean_ll[] = {
  246.3427f, 246.3427f, 492.6854f, 402.4380f, 487.0321f, 128.6856f, 391.6295f,
  281.9928f, 249.9478f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 169.1832f, 402.2611f, 488.1824f, 128.6848f, 407.5836f,
  281.9897f, 249.9489f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f,   0.0000f, 168.6137f,   0.0000f,   0.0000f,   0.0000f,   0.0000f,
    0.0000f};

/* Given 2D and ZRel, we need to find the depth of the root to reconstruct the scale normalized 3D Pose.
   While there exists many 3D poses that can have the same 2D projection, given the 2.5D pose and intrinsic camera parameters,
   there exists a unique 3D pose that satisfies (Xˆn − Xˆm)**2 + (Yˆn − Yˆm)**2 + (Zˆn − Zˆm)**2 = C**2.
   Refer Section 3.3 of https://arxiv.org/pdf/1804.09534.pdf for more details.
*/
std::vector<float> calculateZRoots(const std::vector<float>& X0, const std::vector<float>& X1,
    const std::vector<float>& Y0, const std::vector<float>& Y1,
    const std::vector<float>& Zrel0,
    const std::vector<float>& Zrel1, const std::vector<float>& C) {
    std::vector<float> zRoots(X0.size());
    for (int i = 0; i < X0.size(); i++) {
        double x0 = (double)X0[i], x1 = (double)X1[i], y0 = (double)Y0[i], y1 = (double)Y1[i],
            z0 = (double)Zrel0[i], z1 = (double)Zrel1[i];
        double a = ((x1 - x0) * (x1 - x0)) + ((y1 - y0) * (y1 - y0));
        double b = 2 * (z1 * ((x1 * x1) + (y1 * y1) - x1 * x0 - y1 * y0) +
                    z0 * ((x0 * x0) + (y0 * y0) - x1 * x0 - y1 * y0));
        double c = ((x1 * z1 - x0 * z0) * (x1 * z1 - x0 * z0)) +
                   ((y1 * z1 - y0 * z0) * (y1 * z1 - y0 * z0)) +
                   ((z1 - z0) * (z1 - z0)) - (C[i] * C[i]);
        double d = (b * b) - (4 * a * c);

        // make sure the solutions are valid
        a = fmax(DBL_EPSILON, a);
        d = fmax(DBL_EPSILON, d);
        zRoots[i] = (float) ((-b + sqrt(d)) / (2 * a + 1e-8));
    }
    return zRoots;
}

float median(std::vector<float>& v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

/* Given 2D keypoints and the relative depth of each keypoint w.r.t the root, we find the depth of the root
   to reconstruct the scale normalized 3D pose.
*/
std::vector<NvAR_Point3f> liftKeypoints25DTo3D(const float* p2d,
    const float* pZRel,
    const int numKeypoints,
    const Eigen::Matrix3f& KInv,
    const float limbLengths[]) {

    const int ROOT = 0;

    // Contains the relative depth values of each keypoints
    std::vector<float> zRel(numKeypoints, 0.f);

    // Matrix containing the 2D keypoints.
    Eigen::MatrixXf XY1 = Eigen::MatrixXf(numKeypoints, 3);

    // Mean distance between a specific pair and its parent.
    std::vector<float> C;

    // Indices representing keypoints and its parents for limb lengths > 0.
    // In our dataset, we only have limb length information for few keypoints.
    std::vector<int> idx0 = { 0, 3, 6, 8, 5, 2, 2, 21, 23, 21, 7, 4, 1, 1, 20, 22, 20 };
    std::vector<int> idx1 = { 3, 6, 0, 5, 2, 0, 21, 23, 25, 6, 4, 1, 0, 20, 22, 24, 6 };

    std::vector<float> X0(idx0.size(), 0.f), Y0(idx0.size(), 0.f), X1(idx0.size(), 0.f), Y1(idx0.size(), 0.f),
        zRel0(idx0.size(), 0.f), zRel1(idx0.size(), 0.f);

    for (int i = 0; i < numKeypoints; i++) {
        zRel[i] = pZRel[i];

        XY1.row(i) << p2d[i * 2], p2d[(i * 2) + 1], 1.f;

        if (limbLengths[i] > 0.f) C.push_back(limbLengths[i]);
    }

    // Set relative depth of root to be 0 as the relative depth is measure w.r.t the root.
    zRel[ROOT] = 0.f;

    for (int i = 0; i < XY1.rows(); i++) {
        float x = XY1(i, 0);
        float y = XY1(i, 1);
        float z = XY1(i, 2);
        XY1.row(i) << x, y, z;
    }

    XY1 = XY1 * KInv;

    for (int i = 0; i < idx0.size(); i++) {
        X0[i] = XY1(idx0[i], 0);
        Y0[i] = XY1(idx0[i], 1);
        X1[i] = XY1(idx1[i], 0);
        Y1[i] = XY1(idx1[i], 1);
        zRel0[i] = zRel[idx0[i]];
        zRel1[i] = zRel[idx1[i]];
    }

    std::vector<float> zRoots = calculateZRoots(X0, X1, Y0, Y1, zRel0, zRel1, C);

    float zRootsMedian = median(zRoots);

    zRootsMedian = m_filterRootDepth.filter(zRootsMedian);

    std::vector<NvAR_Point3f> p3d(numKeypoints, { 0.f, 0.f, 0.f });

    for (int i = 0; i < numKeypoints; i++) {
        p3d[i].x = XY1(i, 0) * zRel[i];
        p3d[i].y = XY1(i, 1) * zRel[i];
        p3d[i].z = XY1(i, 2) * zRel[i];
    }

    return p3d;
}

/* Once we have obtained the scale normalized 3D pose, we use the mean limb lengths of keypoint-keypointParent pairs
* to find the scale of the whole body. We solve for
*  s^ = argmin sum((s * L2_norm(P_k - P_l) - meanLimbLength_k_l)**2), solve for s.
*  meanLimbLength_k_l = mean length of the bone between keypoints k and l in the training data
*  P_k and P_l are the keypoint location of k and l.
*
*  We have a least squares minimization, where we are trying to minimize the magnitude of the error:
     (target - scale * unit_length). Thus, we're minimizing T - sL. By the normal equations, the optimal solution is:
      s = inv([L'L]) * L'T

*/
float recoverScale(const std::vector<NvAR_Point3f>& p3d, const float* scores,
    const float targetLengths[]) {
    std::vector<int> validIdx;

    // Indices of keypoints for which we have the length information.
    for (int i = 0; i < p3d.size(); i++) {
        if (targetLengths[i] > 0.f) validIdx.push_back(i);
    }

    Eigen::MatrixXf targetLenMatrix = Eigen::MatrixXf(validIdx.size(), 1);

    for (int i = 0; i < validIdx.size(); i++) {
        targetLenMatrix(i, 0) = targetLengths[validIdx[i]];
    }

    // Indices representing keypoints and its parents for limb lengths > 0.
    // In our dataset, we have only have limb length information for few keypoints.
    std::vector<int> idx0 = { 0, 3, 6, 8, 5, 2, 2, 21, 23, 21, 7, 4, 1, 1, 20, 22, 20 };
    std::vector<int> idx1 = { 3, 6, 0, 5, 2, 0, 21, 23, 25, 6, 4, 1, 0, 20, 22, 24, 6 };

    Eigen::MatrixXf unitLength = Eigen::MatrixXf(idx0.size(), 1);
    Eigen::VectorXf limbScores(unitLength.size());
    float squareNorms = 0.f;
    float limbScoresSum = 0.f;
    for (int i = 0; i < idx0.size(); i++) {
        unitLength(i, 0) = sqrtf((p3d[idx0[i]].x - p3d[idx1[i]].x) * (p3d[idx0[i]].x - p3d[idx1[i]].x) +
            (p3d[idx0[i]].y - p3d[idx1[i]].y) * (p3d[idx0[i]].y - p3d[idx1[i]].y) +
            (p3d[idx0[i]].z - p3d[idx1[i]].z) * (p3d[idx0[i]].z - p3d[idx1[i]].z));

        limbScores[i] = scores[idx0[i]] * scores[idx1[i]];
        limbScoresSum += limbScores[i];
    }

    for (int i = 0; i < limbScores.size(); i++) {
        limbScores[i] /= limbScoresSum;
        squareNorms += ((unitLength(i, 0) * unitLength(i, 0)) * limbScores[i]);
    }

    auto limbScoreDiag = limbScores.asDiagonal();

    //Eigen::MatrixXf numerator1 = ;
    Eigen::MatrixXf numerator = (unitLength.transpose() * limbScoreDiag) * targetLenMatrix;

    return numerator(0, 0) / squareNorms;
}

void osd_upper_body(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3 * _image_width / MUXER_OUTPUT_WIDTH;//6;//3;
  const int keypoint_line_width = 2 * _image_width / MUXER_OUTPUT_WIDTH;//4;//2;

  const int num_joints = 24;
  const int idx_joints[] = { 0,  1,  2,  3,  6, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33};
  const int num_bones = 25;
  const int idx_bones[] = { 21,  6, 20,  6, 21, 23, 20, 22, 24, 22, 23, 25,
                            27, 25, 31, 25, 33, 25, 29, 25, 24, 30, 24, 26,
                            24, 32, 24, 28,  2, 21,  1, 20,  3,  6,  6, 15,
                            15, 16, 15, 17, 19, 17, 18, 16,  0,  1,  0,  2,
                             0,  3};
  const NvOSD_ColorParams bone_colors[] = {
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 1.0, 0, 1}};

  for (int ii = 0; ii < num_joints; ii++) {
    int i = idx_joints[ii];

    if (keypoints_confidence[i] < _sgie_classifier_threshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color =  NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color =  NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < _sgie_classifier_threshold) ||
        (keypoints_confidence[i1] < _sgie_classifier_threshold))
        continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_LineParams *lparams = &dmeta->line_params[dmeta->num_lines];
    lparams->x1 = keypoints[2 * i0];
    lparams->y1 = keypoints[2 * i0 + 1];
    lparams->x2 = keypoints[2 * i1];
    lparams->y2 = keypoints[2 * i1 + 1];
    lparams->line_width = keypoint_line_width;
    lparams->line_color = bone_colors[i];
    dmeta->num_lines++;
  }

  return;
}

void osd_lower_body(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3 * _image_width / MUXER_OUTPUT_WIDTH;//6;//3;
  const int keypoint_line_width = 2 * _image_width / MUXER_OUTPUT_WIDTH;//4;//2;

  const int num_joints = 10;
  const int idx_joints[] = { 4,  5,  7,  8,  9, 10, 11, 12, 13, 14};
  const int num_bones = 10;
  const int idx_bones[] = {  2,  5,  5,  8,  1,  4,  4,  7,  7, 13,
                             8, 14,  8, 10,  7,  9, 11,  9, 12, 10};
  const NvOSD_ColorParams bone_colors[] = {
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{0, 0, 1.0, 1},
                          NvOSD_ColorParams{1.0, 0, 0, 1}};

  for (int ii = 0; ii < num_joints; ii++) {
    int i = idx_joints[ii];

    if (keypoints_confidence[i] < _sgie_classifier_threshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color = NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color = NvOSD_ColorParams{0.96, 0.26, 0.21, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < _sgie_classifier_threshold) ||
        (keypoints_confidence[i1] < _sgie_classifier_threshold))
        continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_LineParams *lparams = &dmeta->line_params[dmeta->num_lines];
    lparams->x1 = keypoints[2 * i0];
    lparams->y1 = keypoints[2 * i0 + 1];
    lparams->x2 = keypoints[2 * i1];
    lparams->y2 = keypoints[2 * i1 + 1];
    lparams->line_width = keypoint_line_width;
    lparams->line_color = bone_colors[i];
    dmeta->num_lines++;
  }

  return;
}

static
gpointer copy_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsJoints *src_joints = (NvDsJoints *) user_meta->user_meta_data;
  NvDsJoints *dst_joints = NULL;

  dst_joints = (NvDsJoints *)g_memdup ((gpointer)src_joints, sizeof(NvDsJoints));
  dst_joints->num_joints = src_joints->num_joints;
  dst_joints->pose_type = src_joints->pose_type;
  dst_joints->joints = (NvDsJoint *)g_memdup ((gpointer)src_joints->joints,
                                    sizeof(NvDsJoint)*src_joints->num_joints);
  return dst_joints;
}

static void
release_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsJoints *src_joints = (NvDsJoints *) user_meta->user_meta_data;
  g_free (src_joints->joints);
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}

void parse_25dpose_from_tensor_meta(NvDsInferTensorMeta *tensor_meta,
      NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
  // const int pelvis = 0;
  // const int left_hip = 1;
  // const int right_hip = 2;
  // const int torso = 3;
  // const int left_knee = 4;
  // const int right_knee = 5;
  // const int neck = 6;
  // const int left_ankle = 7;
  // const int right_ankle = 8;
  // const int left_big_toe = 9;
  // const int right_big_toe = 10;
  // const int left_small_toe = 11;
  // const int right_small_toe = 12;
  // const int left_heel = 13;
  // const int right_heel = 14;
  // const int nose = 15;
  // const int left_eye = 16;
  // const int right_eye = 17;
  // const int left_ear = 18;
  // const int right_ear = 19;
  // const int left_shoulder = 20;
  // const int right_shoulder = 21;
  // const int left_elbow = 22;
  // const int right_elbow = 23;
  // const int left_wrist = 24;
  // const int right_wrist = 25;
  // const int left_pinky_knuckle = 26;
  // const int right_pinky_knuckle = 27;
  // const int left_middle_tip = 28;
  // const int right_middle_tip = 29;
  // const int left_index_knuckle = 30;
  // const int right_index_knuckle = 31;
  // const int left_thumb_tip = 32;
  // const int right_thumb_tip = 33;

  const int numKeyPoints = 34;
  float keypoints[2 * numKeyPoints];
  float keypointsZRel[numKeyPoints];
  float keypoints_confidence[numKeyPoints];

  m_K_inv_transpose = _K.inverse().eval();
  m_K_inv_transpose = m_K_inv_transpose.transpose().eval();

  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  for (unsigned int m=0; m < tensor_meta->num_output_layers;m++){
    NvDsInferLayerInfo *info = &tensor_meta->output_layers_info[m];

    if (!strcmp(info->layerName, "pose25d")) {
      float *data = (float *)tensor_meta->out_buf_ptrs_host[m];
      /*  for (int j =0 ; j < 34; j++){
          printf ("a=%f b=%f c=%f d=%f\n",data[j*4],data[j*4+1],data[j*4+2], data[j*4+3]);
          }*/

      // Initialize
      if (g_filter_pose25d.find(obj_meta->object_id) == g_filter_pose25d.end()) {
        const float m_oneEuroSampleRate = 30.0f;
        // const float m_oneEuroMinCutoffFreq = 0.1f;
        // const float m_oneEuroCutoffSlope = 0.05f;
        const float m_oneEuroDerivCutoffFreq = 1.0f;// Hz

        //std::vector <SF1eFilter*> filter_vec;
        std::vector <OneEuroFilter> filter_vec;

        for (int j=0; j < numKeyPoints*3; j++) {
            //TODO:Pending delete especially when object goes out of view, or ID switch
            //will cause memleak, cleanup required wrap into class
         //   filter_vec.push_back(SF1eFilterCreate(30, 1.0, 0.0, 1.0));

          // filters for x and y
          // for (auto& fil : m_filterKeypoints2D) fil.reset(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq);
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.1f, 0.05, m_oneEuroDerivCutoffFreq));

          // filters for z (depth)
          // for (auto& fil : m_filterKeypointsRelDepth) fil.reset(m_oneEuroSampleRate, 0.5f, 0.05, m_oneEuroDerivCutoffFreq);
          filter_vec.push_back(OneEuroFilter(m_oneEuroSampleRate, 0.5f, 0.05, m_oneEuroDerivCutoffFreq));
        }
        g_filter_pose25d[obj_meta->object_id] = filter_vec;

        // Filters depth of root keypoint
        m_filterRootDepth.reset(m_oneEuroSampleRate, 0.1f, 0.05f, m_oneEuroDerivCutoffFreq);
      }

      int batchSize_offset = 0;

      //std::vector<SF1eFilter*> &filt_val = g_filter_pose25d[obj_meta->object_id];
      std::vector<OneEuroFilter> &filt_val = g_filter_pose25d[obj_meta->object_id];

      // x,y,z,c
      for (int i = 0; i < numKeyPoints; i++) {
        int index = batchSize_offset + i * 4;

        // Update with filtered results
        keypoints[2 * i    ] = filt_val[3 * i    ].filter(data[index    ] *
                                (obj_meta->rect_params.width / 192.0)  + obj_meta->rect_params.left);
        keypoints[2 * i + 1] = filt_val[3 * i + 1].filter(data[index + 1] *
                                (obj_meta->rect_params.height / 256.0) + obj_meta->rect_params.top);
        keypointsZRel[i]     = filt_val[3 * i + 2].filter(data[index + 2]);

        keypoints_confidence[i] = data[index + 3];
      }

      // Since we have cropped and resized the image buffer provided to the SDK from the app,
      // we scale and offset the points back to the original resolution
      float scaleOffsetXY[] = {1.0f, 0.0f, 1.0f, 0.0f};

      // Render upper body
      if (1) {
        osd_upper_body(frame_meta, bmeta, dmeta, numKeyPoints, keypoints, keypoints_confidence);
      }
      // Render lower body
      if (1) {
        osd_lower_body(frame_meta, bmeta, dmeta, numKeyPoints, keypoints, keypoints_confidence);
      }

      // SGIE operates on an enlarged/padded image buffer.
      // const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
      // const int muxer_output_height_pad = _pad_dim * 2 + _image_height;
      // Before outputting result, the image frame with overlay is cropped by removing _pad_dim.
      // The final pose estimation result should counter the padding before deriving 3D keypoints.
      for (int i = 0; i < numKeyPoints; i++) {
        keypoints[2 * i    ]-= _pad_dim;
        keypoints[2 * i + 1]-= _pad_dim;
      }

      // Recover pose 3D
      std::vector<NvAR_Point3f> p3dLifted;
      p3dLifted = liftKeypoints25DTo3D(keypoints, keypointsZRel, numKeyPoints, m_K_inv_transpose, m_scale_ll);
      float scale = recoverScale(p3dLifted, keypoints_confidence, m_mean_ll);
      // printf("scale = %f\n", scale);

      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (bmeta);
      NvDsJoints *ds_joints = (NvDsJoints *)g_malloc(numKeyPoints * sizeof(NvDsJoints));
      ds_joints->num_joints = numKeyPoints;
      ds_joints->pose_type = 1;// datapose3D

      ds_joints->joints = (NvDsJoint *)g_malloc (numKeyPoints * sizeof (NvDsJoint));
        //attach 3D to the user meta data
      for (int i = 0; i < numKeyPoints; i++) {
        ds_joints->joints[i].confidence = keypoints_confidence[i];
        ds_joints->joints[i].x = p3dLifted[i].x;
        ds_joints->joints[i].y = p3dLifted[i].y;
        ds_joints->joints[i].z = p3dLifted[i].z;
        //g_print("%d point [%f, %f, %f]\n", i, p3dLifted[i].x, p3dLifted[i].y, p3dLifted[i].z);
      }
      user_meta->user_meta_data = ds_joints;
      user_meta->base_meta.meta_type = (NvDsMetaType) NVDS_OBJ_META;
      user_meta->base_meta.release_func = release_bodypose_meta;
      user_meta->base_meta.copy_func = copy_bodypose_meta;
      nvds_add_user_meta_to_obj (obj_meta, user_meta);
    }
  }
}

/** Function to get the absolute path of a file.*/
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
        return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else
      return FALSE;
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

NvDsYamlParserStatus
nvds_parse_preprocess (GstElement *element, gchar* app_cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;
  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdspreprocess")) {
    std::cerr << "[ERROR] Passed element is not nvdspreprocess" << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  if (!app_cfg_file_path) {
    printf("Config file not provided.\n");
    return NVDS_YAML_PARSER_ERROR;
  }

  YAML::Node configyml = YAML::LoadFile(app_cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      char *config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            config_file_path)) {
            ret = NVDS_YAML_PARSER_ERROR;
      }
       g_object_set(G_OBJECT(element), "config-file",
         config_file_path, NULL);
         printf("config_file_path:%s\n", config_file_path);
      g_free (str);
      g_free(config_file_path);
    } else {
      printf("[WARNING] Unknown param found in preprocess: %s\n", paramKey.c_str());
    }
  }
  return ret;
}


static void
parse_streammux_width_height_yaml (gint *width, gint *height, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["streammux"].begin();
     itr != configyml["streammux"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "width") {
      *width = itr->second.as<gint>();
    } else if(paramKey == "height"){
      *height = itr->second.as<gint>();
    }
  }
}

static void
parse_sink_type_yaml (gint *type, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["sink"].begin();
     itr != configyml["sink"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "sink-type") {
      *type = itr->second.as<gint>();
    }
  }
}

static void
parse_sink_enc_type_yaml (gint *enc_type, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["sink"].begin();
     itr != configyml["sink"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enc-type") {
       int value = itr->second.as<gint>();
       if(value == 0 || value == 1){
        *enc_type = value;
       }
    }
  }
}

/* pgie_src_pad_buffer_probe will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      float sizex = obj_meta->rect_params.width * .5f;
      float sizey = obj_meta->rect_params.height * .5f;
      float centrx = obj_meta->rect_params.left  + sizex;
      float centry = obj_meta->rect_params.top  + sizey;
      sizex *= (1.25f);
      sizey *= (1.25f);
      if (sizex < sizey)
        sizex = sizey;
      else
        sizey = sizex;

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);

      sizex= obj_meta->rect_params.width * .5f, sizey = obj_meta->rect_params.height * .5f;
      centrx = obj_meta->rect_params.left + sizex, centry = obj_meta->rect_params.top + sizey;
      // Make sure box has same aspect ratio as 3D Body Pose model's input dimensions
      // (e.g 192x256 -> 0.75 aspect ratio) by enlarging in the appropriate dimension.
      float xScale = (float)192.0 / (float)sizex, yScale = (float)256.0 / (float)sizey;
      if (xScale < yScale) { // expand on height
          sizey = (float)256.0/ xScale;
      }
      else { // expand on width
          sizex = (float)192.0 / yScale;
      }

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);
      if (obj_meta->rect_params.left < 0.0) {
          obj_meta->rect_params.left = 0.0;
      }
      if (obj_meta->rect_params.top < 0.0) {
        obj_meta->rect_params.top = 0.0;
      }
      if (obj_meta->rect_params.left + obj_meta->rect_params.width > muxer_output_width_pad -1){
        obj_meta->rect_params.width = muxer_output_width_pad - 1 - obj_meta->rect_params.left;
      }
      if (obj_meta->rect_params.top + obj_meta->rect_params.height > muxer_output_height_pad -1){
        obj_meta->rect_params.height = muxer_output_height_pad - 1 - obj_meta->rect_params.top;
      }

    }
  }
  return GST_PAD_PROBE_OK;
}

/* sgie_src_pad_buffer_probe  will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
sgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  // g_mutex_lock(&str->struct_lock);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      if(obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        // Set below values to 0 in order to disable bbox and text output
        obj_meta->rect_params.border_width = 2;
        obj_meta->text_params.font_params.font_size = 10;

        for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
             l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
          if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
          {
            NvDsInferTensorMeta *tensor_meta =
                (NvDsInferTensorMeta *)user_meta->user_meta_data;
            parse_25dpose_from_tensor_meta(tensor_meta, frame_meta, obj_meta) ;
          }
        }
      }
    }
  }

  return GST_PAD_PROBE_OK;
}

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

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, " ");

    txt_params->x_offset = 100;
    txt_params->y_offset = 200;

    char font_name[] = "Mono";
    txt_params->font_params.font_name = font_name;
    txt_params->font_params.font_size = 15;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

typedef struct _DsSourceBin
{
    GstElement *source_bin;
    GstElement *uri_decode_bin;
    GstElement *vidconv;
    GstElement *nvvidconv;
    GstElement *capsfilt;
    GstElement *capsraw;
    gint index;
}DsSourceBinStruct;

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  DsSourceBinStruct *bin_struct = (DsSourceBinStruct *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad to videoconvert if no hardware decoder is used */
    if (bin_struct->vidconv) {
      GstPad *conv_sink_pad = gst_element_get_static_pad (bin_struct->vidconv,
          "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad)) {
        g_printerr ("Failed to link decoderbin src pad to"
            " converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
      if (!gst_element_link_many (bin_struct->vidconv, bin_struct->capsraw,
         bin_struct->nvvidconv, NULL)) {
         g_printerr ("Failed to link videoconvert to nvvideoconvert\n");
      }
    } else {
      GstPad *conv_sink_pad = gst_element_get_static_pad (bin_struct->nvvidconv,
          "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad)) {
        g_printerr ("Failed to link decoderbin src pad to "
            "converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
    }
    if (gst_caps_features_contains (features, "memory:NVMM")) {
      g_print ("###Decodebin pick nvidia decoder plugin.\n");
    } else {
      /* Get the source bin ghost pad */
      g_print ("###Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  DsSourceBinStruct *bin_struct = (DsSourceBinStruct *) user_data;
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strstr_len (name, -1, "pngdec") == name) {
    bin_struct->vidconv = gst_element_factory_make ("videoconvert",
        "source_vidconv");
    bin_struct->capsraw = gst_element_factory_make ("capsfilter",
        "raw_caps");
    GstCaps *caps = gst_caps_new_simple ("video/x-raw", "format",
        G_TYPE_STRING, "I420", NULL);
    g_object_set (G_OBJECT (bin_struct->capsraw), "caps", caps, NULL);
    gst_bin_add_many (GST_BIN (bin_struct->source_bin), bin_struct->vidconv,
      bin_struct->capsraw, NULL);
  } else {
    bin_struct->vidconv = NULL;
  }
}

static bool
create_source_bin (DsSourceBinStruct *ds_source_struct, gchar * uri)
{
  gchar bin_name[16] = { };
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  ds_source_struct->nvvidconv = NULL;
  ds_source_struct->capsfilt = NULL;
  ds_source_struct->source_bin = NULL;
  ds_source_struct->uri_decode_bin = NULL;

  g_snprintf (bin_name, 15, "source-bin-%02d", ds_source_struct->index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  ds_source_struct->source_bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  ds_source_struct->uri_decode_bin = gst_element_factory_make ("uridecodebin",
      "uri-decode-bin");
  ds_source_struct->nvvidconv = gst_element_factory_make ("nvvideoconvert",
      "source_nvvidconv");
  ds_source_struct->capsfilt = gst_element_factory_make ("capsfilter",
      "source_capset");

  if (!ds_source_struct->source_bin || !ds_source_struct->uri_decode_bin ||
      !ds_source_struct->nvvidconv
      || !ds_source_struct->capsfilt) {
    g_printerr ("One element in source bin could not be created.\n");
    return false;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (ds_source_struct->uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (ds_source_struct->uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), ds_source_struct);
  g_signal_connect (G_OBJECT (ds_source_struct->uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), ds_source_struct);

  caps = gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "NV12",
      NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (ds_source_struct->capsfilt), "caps", caps, NULL);

  gst_bin_add_many (GST_BIN (ds_source_struct->source_bin),
      ds_source_struct->uri_decode_bin, ds_source_struct->nvvidconv,
      ds_source_struct->capsfilt, NULL);

  if (!gst_element_link (ds_source_struct->nvvidconv,
      ds_source_struct->capsfilt)) {
    g_printerr ("Could not link vidconv and capsfilter\n");
    return false;
  }

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  GstPad *gstpad = gst_element_get_static_pad (ds_source_struct->capsfilt,
      "src");
  if (!gstpad) {
    g_printerr ("Could not find srcpad in '%s'",
        GST_ELEMENT_NAME(ds_source_struct->capsfilt));
      return false;
  }
  if(!gst_element_add_pad (ds_source_struct->source_bin,
      gst_ghost_pad_new ("src", gstpad))) {
    g_printerr ("Could not add ghost pad in '%s'",
        GST_ELEMENT_NAME(ds_source_struct->capsfilt));
  }
  gst_object_unref (gstpad);

  return true;
}

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler (int signum)
{
  struct sigaction action;

  NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  _cintr = TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (_quit) {
    return FALSE;
  }

  if (_cintr) {
    _cintr = FALSE;

    _quit = TRUE;
    GMainLoop *loop = (GMainLoop *) data;
    g_main_loop_quit (loop);

    return FALSE;
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
  guint num_sources = 0;

  GMainLoop *loop = NULL;
  GstCaps *caps = NULL;
  GstElement *streammux = NULL, *pgie = NULL, *sgie = NULL, *preprocess1 = NULL, *sgie1 = NULL;;
  // Padding the image and removing the padding
  GstElement *nvvideoconvert_enlarge = NULL, *nvvideoconvert_reduce = NULL,
    *capsFilter_enlarge = NULL, *capsFilter_reduce = NULL;
  GstElement *nvvidconv = NULL, *nvtile = NULL, *nvosd = NULL, *tracker = NULL, *nvdslogger = NULL;
  GstElement *sink = NULL;
  DsSourceBinStruct source_struct[128];
  GstBus *bus = NULL;
  guint bus_watch_id;

  gboolean useDisplay = FALSE;
  gboolean useFakeSink = FALSE;
  gboolean useFileSink = FALSE;
  guint tiler_rows, tiler_columns;
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isStreaming=false;
  GList* g_list = NULL;
  GList* iterator = NULL;
  bool isH264 = true;
  gchar *filepath = NULL;


  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Standard GStreamer initialization */
  // signal(SIGINT, sigintHandler);
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, NULL);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream_pose_classfication_app");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "streammux-pgie");
  if (!streammux) {
    g_printerr ("PGIE streammux could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  parse_streammux_width_height_yaml(&_image_width, &_image_height, argv[1]);
  g_print("width %d hight %d\n", _image_width, _image_height);

  _focal_length = (float)_focal_length_dbl;
  if (_focal_length <= 0) {
    g_printerr("--focal value %f is non-positive. Exiting...\n", _focal_length);
    return false;
  }

  _K.row(0) << _focal_length, 0,              _image_width / 2.f;
  _K.row(1) << 0,             _focal_length,  _image_height / 2.f;
  _K.row(2) << 0,             0,              1.f;

  _pad_dim = PAD_DIM * _image_width / MUXER_OUTPUT_WIDTH;
  //---Set properties of streammux---

  if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
    g_printerr ("No source is found. Exiting.\n");
    return -1;
  }

  for (iterator = g_list, num_sources=0; iterator; iterator = iterator->next,num_sources++) {
    /* Source element for reading from the file */
    source_struct[num_sources].index = num_sources;

    if (g_strrstr ((gchar *)iterator->data, "rtsp://") ||
        g_strrstr ((gchar *)iterator->data, "v4l2://") ||
        g_strrstr ((gchar *)iterator->data, "http://") ||
        g_strrstr ((gchar *)iterator->data, "rtmp://")) {
      isStreaming = true;
    } else {
      isStreaming = false;
    }

    g_print("video %s\n", (gchar *)iterator->data);

    if (!create_source_bin (&(source_struct[num_sources]), (gchar *)iterator->data))
    {
      g_printerr ("Source bin could not be created. Exiting.\n");
      return -1;
    }
      
    gst_bin_add (GST_BIN (pipeline), source_struct[num_sources].source_bin);
      
    g_snprintf (pad_name_sink, 64, "sink_%d", num_sources);
    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_struct[num_sources].source_bin,
        pad_name_src);
    if (!srcpad) {
      g_printerr ("Decoder request src pad failed. Exiting.\n");
      return -1;
    }
    GstPadLinkReturn ret = gst_pad_link (srcpad, sinkpad);
    if ( ret != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting. %d\n",ret);
      return -1;
    }
    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
  }

  nvds_parse_streammux(streammux, argv[1], "streammux");

  if (isStreaming)
    g_object_set (G_OBJECT (streammux), "live-source", true, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      pgie = gst_element_factory_make("nvinferserver", "primary-nvinference-engine");
  } else {
      pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  }
  if (!pgie) {
    g_printerr ("PGIE element could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_gie (pgie, argv[1], "primary-gie");

  /* Override the batch-size set in the config file with the number of sources. */
  guint pgie_batch_size = 0;
  g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);

    g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
  }

  //---Set pgie properties---

  /* We need to have a tracker to track the identified objects */
  tracker = gst_element_factory_make ("nvtracker", "tracker");
  if (!tracker) {
    g_printerr ("Nvtracker could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_tracker(tracker, argv[1], "tracker");

  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
  if (!nvdslogger) {
      g_printerr ("Nvdslogger could not be created. Exiting.\n");
      return -1;
  }
  g_object_set (G_OBJECT(nvdslogger), "fps-measurement-interval-sec",
        1, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* pgie_src_pad = gst_element_get_static_pad(tracker, "src");
  if (!pgie_src_pad)
    g_printerr ("Unable to get src pad for pgie\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (pgie_src_pad);

  /* 3d bodypose secondary gie */
  NvDsGieType sgie0_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie0_type, argv[1], "secondary-gie0"));
  if (sgie0_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      sgie = gst_element_factory_make("nvinferserver", "secondary-nvinference-engine");
  } else {
      sgie = gst_element_factory_make("nvinfer", "secondary-nvinference-engine");
  }
  if (!sgie) {
    g_printerr ("Secondary nvinfer could not be created. Exiting.\n");
    return -1;
  }
  //---Set sgie properties---
  /* Configure the nvinfer element using the nvinfer config file. */
  nvds_parse_gie (sgie, argv[1], "secondary-gie0");

  /* Override the batch-size set in the config file with the number of sources. */
  guint sgie_batch_size = 0;
  g_object_get(G_OBJECT(sgie), "batch-size", &sgie_batch_size, NULL);
  if (sgie_batch_size < num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        sgie_batch_size, num_sources);

    g_object_set(G_OBJECT(sgie), "batch-size", num_sources, NULL);
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* sgie_src_pad = gst_element_get_static_pad(sgie, "src");
  if (!sgie_src_pad)
    g_printerr("Unable to get src pad for sgie\n");
  else
    gst_pad_add_probe(sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        sgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(sgie_src_pad);
  //---Set sgie properties---

  /* preprocess + bodypose classification */
  preprocess1 = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");
  nvds_parse_preprocess(preprocess1, argv[1], "secondary-preprocess1");
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie1_type, argv[1], "secondary-gie1"));
  if (sgie1_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      sgie1 = gst_element_factory_make("nvinferserver", "bodypose-classification-nvinference-engine");
  } else {
      sgie1 = gst_element_factory_make("nvinfer", "bodypose-classification-nvinference-engine");
  }
  if (!sgie1) {
    g_printerr ("sgie1 could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_gie (sgie1, argv[1], "secondary-gie1");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  if (!nvvidconv) {
    g_printerr ("nvvidconv could not be created. Exiting.\n");
    return -1;
  }

  //---Manipulate image size so that PGIE bbox is large enough---
  // Enlarge image so that PeopleNet detected bbox is larger which would fully cover the
  // detected object in the original sized image.
  nvvideoconvert_enlarge = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_enlarge");
  if (!nvvideoconvert_enlarge) {
    g_printerr ("nvvideoconvert_enlarge could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_enlarge = gst_element_factory_make("capsfilter", "capsFilter_enlarge");
  if (!capsFilter_enlarge) {
    g_printerr ("capsFilter_enlarge could not be created. Exiting.\n");
    return -1;
  }

  // Reduce the previously enlarged image frame so that the final output video retains the
  // same dimension as the pipeline's input video dimension.
  nvvideoconvert_reduce = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_reduce");
  if (!nvvideoconvert_reduce) {
    g_printerr ("nvvideoconvert_reduce could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_reduce = gst_element_factory_make("capsfilter", "capsFilter_reduce");
  if (!capsFilter_reduce) {
    g_printerr ("capsFilter_reduce could not be created. Exiting.\n");
    return -1;
  }

  gchar *string1 = NULL;
  asprintf (&string1, "%d:%d:%d:%d", _pad_dim, _pad_dim, _image_width, _image_height);
  // "dest-crop" - input size < output size
  g_object_set(G_OBJECT(nvvideoconvert_enlarge), "dest-crop", string1,"interpolation-method",1 ,NULL);
  // "src-crop" - input size > output size
  g_object_set(G_OBJECT(nvvideoconvert_reduce), "src-crop", string1,"interpolation-method",1 ,NULL);
  free(string1);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;
  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      muxer_output_width_pad, muxer_output_height_pad);
  GstCaps *caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_enlarge),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);

  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      _image_width, _image_height);
  caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_reduce),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);
  //---Manipulate image size so that PGIE bbox is large enough---

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("Nvdsosd could not be created. Exiting.\n");
    return -1;
  }
  nvtile = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", 1280, "height", 720, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref(osd_sink_pad);

  /* Set output file location */
  int sink_type = 0;
  parse_sink_type_yaml(&sink_type, argv[1]);
  int enc_type = 0;
  parse_sink_enc_type_yaml(&enc_type, argv[1]);
  g_print("sink_type:%d, enc_type:%d\n", sink_type, enc_type);

  if(sink_type == 1) {
    sink = gst_element_factory_make("nvvideoencfilesinkbin", "nv-filesink");
    if (!sink) {
      g_printerr ("Filesink could not be created. Exiting.\n");
      return -1;
    }
    g_object_set(G_OBJECT(sink), "output-file", "out.mp4", NULL);
    g_object_set(G_OBJECT(sink), "bitrate", 4000000, NULL);
    //g_object_set(G_OBJECT(sink), "profile", 3, NULL);
    g_object_set(G_OBJECT(sink), "codec", 1, NULL);//hevc
    // g_object_set(G_OBJECT(sink), "control-rate", 0, NULL);//hevc
    g_object_set(G_OBJECT(sink), "enc-type", enc_type, NULL);
  } else if(sink_type == 2) {
    sink = gst_element_factory_make("nvrtspoutsinkbin", "nv-rtspsink");
    if (!sink) {
      g_printerr ("Filesink could not be created. Exiting.\n");
      return -1;
    }
    g_object_set(G_OBJECT(sink), "enc-type", enc_type, NULL);
  } else if(sink_type == 3) {
    if (prop.integrated) {
      sink = gst_element_factory_make("nv3dsink", "nv-sink");
    } else {
#ifdef __aarch64__
      sink = gst_element_factory_make("nv3dsink", "nv-sink");
#else
      sink = gst_element_factory_make("nveglglessink", "nv-sink");
#endif
    }
  } else {
    sink = gst_element_factory_make("fakesink", "nv-fakesink");
  }

  /* Add all elements to the pipeline */
  // streammux has been added into pipeline already.
  gst_bin_add_many(GST_BIN(pipeline),
    nvvideoconvert_enlarge, capsFilter_enlarge,
    pgie, tracker, sgie, preprocess1, sgie1, nvtile,
    nvvidconv, nvosd, sink, nvdslogger,
    nvvideoconvert_reduce, capsFilter_reduce, NULL);

  // Link elements
  if (!gst_element_link_many(streammux,
      nvvideoconvert_enlarge, capsFilter_enlarge, pgie, tracker, sgie, preprocess1, sgie1,
      nvdslogger, nvvideoconvert_reduce, capsFilter_reduce, nvtile, nvvidconv, nvosd,  sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Set the pipeline to "playing" state */
  g_print("Now playing!\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  GST_DEBUG_BIN_TO_DOT_FILE((GstBin*)pipeline, GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;

}
