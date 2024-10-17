/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

#ifndef _DS_YAML_PARSER_H_
#define _DS_YAML_PARSER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include "nvds_yml_parser.h"

typedef enum
{
  ENCODER_TYPE_HW,
  ENCODER_TYPE_SW
} EncHwSwType;

NvDsYamlParserStatus
ds_parse_rtsp_output(GstElement *sink,
  GstRTSPServer *server, GstRTSPMediaFactory *factory,
  gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_enc_config(GstElement *encoder, 
  gchar *cfg_file_path, const char* group);

guint
ds_parse_group_type(gchar *cfg_file_path, const char* group);

guint
ds_parse_enc_type(gchar *cfg_file_path, const char* group);

guint
ds_parse_enc_codec(gchar *cfg_file_path, const char* group);

GString *
ds_parse_file_name(gchar *cfg_file_path, const char* group);

GString *
ds_parse_config_yml_filepath(gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group);

NvDsYamlParserStatus
ds_parse_ocdr_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group);

void
create_video_encoder(bool isH264, int enc_type, GstElement** conv_capfilter,
  GstElement** outenc, GstElement** encparse, GstElement** rtppay);

guint
ds_parse_group_enable(gchar *cfg_file_path, const char* group);
guint
ds_parse_group_car_mode(gchar *cfg_file_path, const char* group);

void
get_triton_yml(gint car_mode, gboolean use_triton_grpc, char* pgie_cfg_file_path,
char* sgie1_cfg_file_path, char* sgie2_cfg_file_path, guint buf_len);

#ifdef __cplusplus
}
#endif

#endif
