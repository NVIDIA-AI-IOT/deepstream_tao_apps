/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef _NVDS_FACIAL_META_H_
#define _NVDS_FACIAL_META_H_

#include <gst/gst.h>
#include <glib.h>
#include "nvdsmeta.h"
#include "cv/faciallandmarks/FacialLandmarks.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define NVDS_USER_RIVA_META_FACEMARK (nvds_get_user_meta_type((gchar*)"NVIDIA.RIVA.USER_META_FACEMARK"))
#define FACEMARK_TOTAL_NUM 200

typedef struct{
    float x;
    float y;
    float score;
}FacePoint;

typedef struct {
    float left;
    float top;
    float right;
    float bottom;

    float width() { return right - left; }
    float height() { return bottom - top; }
    float center_x() { return left + width() / 2; }
    float center_y() { return right + height() / 2; }
}BBox;

typedef struct
{
  FacePoint mark[FACEMARK_TOTAL_NUM];
  BBox left_eye_rect;
  BBox right_eye_rect;
  int facemark_num;
}NvDsFacePointsMetaData;

gboolean nvds_add_facemark_meta (NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta,
         cvcore::ArrayN<cvcore::Vector2f, cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS> &marks,
         float *confidence);

#ifdef __cplusplus
}
#endif

#endif
