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

/**
 * @file nvdspreprocess_lib.h
 * <b>NVIDIA DeepStream Preprocess lib specifications </b>
 *
 * @b Description: This file defines common elements used in the API
 * exposed by the Gst-nvdspreprocess plugin.
 */

/**
 * @defgroup  gstreamer_nvdspreprocess_api NvDsPreProcess Plugin
 * Defines an API for the GStreamer NvDsPreProcess custom lib.
 * @ingroup custom_gstreamer
 * @{
 */

#ifndef __NVDSPREPROCESS_LIB__
#define __NVDSPREPROCESS_LIB__

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"

/** Maximum file path length */
#define _PATH_MAX 4096

/** actual frames sequence length */
#define NVDSPREPROCESS_USER_CONFIGS_FRAMES_SEQUENCE_LENGHTH "frames-sequence-length"

/**
 * Custom transformation function for group
 */
extern "C"
NvDsPreProcessStatus CustomTransformation(NvBufSurface *in_surf,
                                          NvBufSurface *out_surf,
                                          CustomTransformParams &params);

/**
 * Custom Asynchronus group transformation function
 */
extern "C"
NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf,
                                          NvBufSurface *out_surf,
                                          CustomTransformParams &params);

/**
 * Custom tensor preparation function for NCHW/NHWC network order
 */
extern "C"
NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch, NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam, NvDsPreProcessAcquirer *acquirer);

/**
 * custom library initialization function
 */
extern "C"
CustomCtx *initLib(CustomInitParams initparams);

/**
 * custom library deinitialization function
 */
extern "C"
void deInitLib(CustomCtx *ctx);

#endif
