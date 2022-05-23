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
#ifndef _NVDS_GAZE_META_H_
#define _NVDS_GAZE_META_H_

#include <gst/gst.h>
#include <glib.h>
#include "nvdsmeta.h"
#include "cv/gazenet/GazeNet.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define NVDS_USER_RIVA_META_GAZE (nvds_get_user_meta_type((gchar*)"NVIDIA.RIVA.USER_META_GAZE"))

typedef struct
{
  float gaze_params[cvcore::gazenet::GazeNet::OUTPUT_SIZE];
  int left_start_x;
  int left_start_y;
  int left_end_x;
  int left_end_y;
  int right_start_x;
  int right_start_y;
  int right_end_x;
  int right_end_y;
}NvDsGazeMetaData;

gboolean nvds_add_gaze_meta (NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, 
        cvcore::ArrayN<float, cvcore::gazenet::GazeNet::OUTPUT_SIZE> &params,
        cvcore::Array<cvcore::Vector2i> &leftStart,
        cvcore::Array<cvcore::Vector2i> &leftEnd,
        cvcore::Array<cvcore::Vector2i> &rightStart,
        cvcore::Array<cvcore::Vector2i> &rightEnd);

#ifdef __cplusplus
}
#endif

#endif
