/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "ds_gaze_meta.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include "gstnvdsmeta.h"

/*GazeNet metadata functions*/

static
gpointer nvds_copy_gaze_meta (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsGazeMetaData *p_gaze_meta_data = (NvDsGazeMetaData *) user_meta->user_meta_data;
    NvDsGazeMetaData *pnew_gaze_meta_data = (NvDsGazeMetaData *)g_memdup ( p_gaze_meta_data, sizeof(NvDsGazeMetaData));
    return (gpointer) pnew_gaze_meta_data;
}

static
void nvds_release_gaze_meta (gpointer data,  gpointer user_data)
{
    NvDsUserMeta* user_meta = (NvDsUserMeta*)data;
    NvDsGazeMetaData *p_gaze_meta_data = (NvDsGazeMetaData *) user_meta->user_meta_data;
    delete p_gaze_meta_data;
}

/* Gazenet model outputs 5 float parameters */
extern "C"
gboolean nvds_add_gaze_meta (NvDsBatchMeta *batch_meta, NvDsObjectMeta *obj_meta, 
        cvcore::ArrayN<float, cvcore::gazenet::GazeNet::OUTPUT_SIZE> &params)
{
    NvDsUserMeta *user_meta = NULL;
    user_meta = nvds_acquire_user_meta_from_pool (batch_meta);
    NvDsMetaType user_meta_type = (NvDsMetaType) NVDS_USER_JARVIS_META_GAZE;
    NvDsGazeMetaData *p_gaze_meta_data = new NvDsGazeMetaData;
    
    int params_num = params.getSize();
    
    for (int n = 0; n < params_num; n++) {
        p_gaze_meta_data->gaze_params[n] = params[n];
    }

    user_meta->user_meta_data = (void *) (p_gaze_meta_data);
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func =
        (NvDsMetaCopyFunc) nvds_copy_gaze_meta;
    user_meta->base_meta.release_func =
        (NvDsMetaReleaseFunc) nvds_release_gaze_meta;

    /* We want to add NvDsUserMeta to obj level */
    nvds_add_user_meta_to_obj (obj_meta, user_meta);
    return true;
}
