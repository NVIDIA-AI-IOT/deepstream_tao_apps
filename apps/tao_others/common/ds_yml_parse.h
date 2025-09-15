/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
#define _PATH_MAX 1024

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

NvDsYamlParserStatus
ds_parse_nvdsanalytics(GstElement *element, gchar *cfg_file_path, const char* group);

void
create_video_encoder(bool isH264, int enc_type, GstElement** conv_capfilter,
  GstElement** outenc, GstElement** encparse, GstElement** rtppay);

  /** Function to get the absolute path of a file.*/
gboolean
get_absolute_file_path_yaml (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str);

/** Parse preprocess configurations. */
NvDsYamlParserStatus
nvds_parse_preprocess (GstElement *element, gchar* app_cfg_file_path, const char* group);

/** Parse postprocess configurations. */
NvDsYamlParserStatus
nvds_parse_postprocess (GstElement *element, gchar* app_cfg_file_path, const char* group);

/** Parse width and height of nvstreammux. */
void
parse_streammux_width_height_yaml (gint *width, gint *height, gchar *cfg_file_path);

/** Parse type of sink. */
void
parse_sink_type_yaml (gint *type, gchar *cfg_file_path);

/** Parse enc type of sink. */
void
parse_sink_enc_type_yaml (gint *enc_type, gchar *cfg_file_path);

#ifdef __cplusplus
}
#endif

#endif
