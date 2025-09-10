/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "post_processor_bodypose.h"
#include <algorithm>
#include <Eigen/Dense>

using namespace std;
#define MAX_TIME_STAMP_LEN 32

Eigen::Matrix3f _K;// Camera intrinsic matrix
//===Global variables===
std::unordered_map<gint , std::vector<OneEuroFilter>> g_filter_pose25d;
OneEuroFilter m_filterRootDepth; // Root node in pose25d.
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

// Default camera attributes
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720
#define FOCAL_LENGTH 800.79041f
double _focal_length_dbl = FOCAL_LENGTH;
float _focal_length = (float)_focal_length_dbl;
int _image_width = MUXER_OUTPUT_WIDTH;
int _image_height = MUXER_OUTPUT_HEIGHT;
static float _sgie_classifier_threshold = FLT_MIN;
/* Padding due to AR SDK model requires bigger bboxes*/
#define PAD_DIM 128
int _pad_dim = PAD_DIM;// A scaled version of PAD_DIM

#define ACQUIRE_DISP_META(dmeta)  \
  if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META  || \
      dmeta->num_labels == MAX_ELEMENTS_IN_DISPLAY_META ||  \
      dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) \
        { \
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);\
          nvds_add_display_meta_to_frame(frame_meta, dmeta);\
        }\

#define GET_LINE(lparams) \
        ACQUIRE_DISP_META(dmeta)\
        lparams = &dmeta->line_params[dmeta->num_lines];\
        dmeta->num_lines++;\

static
void generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME,  &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec/1000000;
  g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static
gpointer copy_bodypose_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsJoints *src_joints = (NvDsJoints *) user_meta->user_meta_data;
  NvDsJoints *dst_joints = NULL;

  dst_joints = (NvDsJoints *)g_memdup2 ((gpointer)src_joints, sizeof(NvDsJoints));
  dst_joints->num_joints = src_joints->num_joints;
  dst_joints->pose_type = src_joints->pose_type;
  dst_joints->joints = (NvDsJoint *)g_memdup2 ((gpointer)src_joints->joints,
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

NvDsPostProcessStatus
BodyPoseModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{
  ModelPostProcessor::initResource(initParams);
  m_ClassificationThreshold = initParams.classifierThreshold;
  _K.row(0) << _focal_length, 0,              _image_width / 2.f;
  _K.row(1) << 0,             _focal_length,  _image_height / 2.f;
  _K.row(2) << 0,             0,              1.f;

  _pad_dim = PAD_DIM * _image_width / MUXER_OUTPUT_WIDTH;
  return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
BodyPoseModelPostProcessor::parseEachFrame(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessFrameOutput& result)
{
    result.outputType = NvDsPostProcessNetworkType_BodyPose;
    fillBodyPoseOutput(outputLayers, result.bodyPoseOutput);
    return NVDSPOSTPROCESS_SUCCESS;
}

NvDsPostProcessStatus
BodyPoseModelPostProcessor::fillBodyPoseOutput(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output)
{
    movenetposeFromTensorMeta(outputLayers, output);
    return NVDSPOSTPROCESS_SUCCESS;
}

typedef struct NvAR_Point3f {
  float x, y, z;
} NvAR_Point3f;

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

/*  redundant logic
    for (int i = 0; i < XY1.rows(); i++) {
        float x = XY1(i, 0);
        float y = XY1(i, 1);
        float z = XY1(i, 2);
        XY1.row(i) << x, y, z;
    }
*/
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
      // for (int j =0 ; j < 34; j++) {
      //   printf ("a=%f b=%f c=%f d=%f\n",data[j*4],data[j*4+1],data[j*4+2], data[j*4+3]);
      // }

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
      // float scale = recoverScale(p3dLifted, keypoints_confidence, m_mean_ll);
      // printf("scale = %f\n", scale);
      // for (auto i = 0; i < p3dLifted.size(); i++) {
      //   p3dLifted[i].x *= scale;
      //   p3dLifted[i].y *= scale;
      //   p3dLifted[i].z *= scale;
      // }

      NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (bmeta);
      NvDsJoints *ds_joints = (NvDsJoints *)g_malloc(sizeof(NvDsJoints));
      ds_joints->num_joints = numKeyPoints;
      ds_joints->pose_type = 1;// datapose3D
      ds_joints->joints = (NvDsJoint *)g_malloc (numKeyPoints * sizeof(NvDsJoint));

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

void
BodyPoseModelPostProcessor::attachMetadata     (NvBufSurface *surf, gint batch_idx,
    NvDsBatchMeta  *batch_meta,
    NvDsFrameMeta  *frame_meta,
    NvDsObjectMeta  *obj_meta,
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
{

}

void
BodyPoseModelPostProcessor::prcoessMetadata (NvDsInferTensorMeta *tensor_meta,
    NvDsFrameMeta *frame_meta, NvDsObjectMeta *obj_meta)
{
    parse_25dpose_from_tensor_meta(tensor_meta, frame_meta, obj_meta);
}

void
BodyPoseModelPostProcessor::releaseFrameOutput(NvDsPostProcessFrameOutput& frameOutput)
{
    switch (frameOutput.outputType)
    {
        case NvDsPostProcessNetworkType_BodyPose:
          //Release if meta not attached
            //delete[] frameOutput.segmentationOutput.class_map;
            break;
        default:
            break;
    }
}

float BodyPoseModelPostProcessor::median(std::vector<float>& v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

void BodyPoseModelPostProcessor::osdBody(NvDsFrameMeta* frame_meta,
      NvDsBatchMeta *bmeta,
      NvDsDisplayMeta *dmeta,
      const int numKeyPoints,
      const float keypoints[],
      const float keypoints_confidence[])
{
  const int keypoint_radius = 3;//6;//3;
  const int keypoint_line_width = 2;//4;//2;

  const int num_joints = 17;
  const int num_bones = 18;
  const int idx_bones[] = { 0,1, 0,2, 1,3, 2,4,
                            0,5, 0,6, 5,6, 5,7,
                            7,9, 6,8, 8,10, 11,12,
                            5,11, 11,13, 13,15, 6,12,
                            12,14, 14,16};
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
    int i = ii;// idx_joints[ii];


    if (keypoints_confidence[i] < m_ClassificationThreshold)
      continue;

    ACQUIRE_DISP_META(dmeta);
    NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
    cparams.xc = keypoints[2 * i    ];
    cparams.yc = keypoints[2 * i + 1];
    cparams.radius = keypoint_radius;
    cparams.circle_color = NvOSD_ColorParams{1.0, 0, 0, 1};
    cparams.has_bg_color = 1;
    cparams.bg_color = NvOSD_ColorParams{1.0, 0, 0, 1};
    dmeta->num_circles++;
  }

  for (int i = 0; i < num_bones; i++) {
    int i0 = idx_bones[2 * i    ];
    int i1 = idx_bones[2 * i + 1];

    if ((keypoints_confidence[i0] < m_ClassificationThreshold) ||
        (keypoints_confidence[i1] < m_ClassificationThreshold))
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


void BodyPoseModelPostProcessor::movenetposeFromTensorMeta(
    const std::vector<NvDsInferLayerInfo>& outputLayers,
    NvDsPostProcessBodyPoseOutput& output)
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

  unsigned int numAttributes = outputLayers.size();
  for (unsigned int m=0; m < numAttributes;m++){
    const NvDsInferLayerInfo *info = &outputLayers[m];
    if (!strcmp(info->layerName, "output_0")) {
      output.data = (float *)info->buffer;

    }
  }
}


