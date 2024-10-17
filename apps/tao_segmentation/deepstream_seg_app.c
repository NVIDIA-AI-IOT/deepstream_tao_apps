/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <string.h>
#include <sys/time.h>
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvds_yml_parser.h"
#include "cuda_runtime_api.h"


/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/*segvisual's dimention must be greater than or equal to model's.
 read model's dimention from cfg, or use this default value*/
#define SEG_OUTPUT_WIDTH 1920
#define SEG_OUTPUT_HEIGHT 1080

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

#define MAX_SOURCE_BINS 1024

static guint fileLoop;
static guint networkType;
static guint numDetectedClasses;

typedef struct
{
  gdouble fps[MAX_SOURCE_BINS];
  gdouble fps_avg[MAX_SOURCE_BINS];
  guint num_instances;
}PerfStruct;

typedef struct
{
  guint buffer_cnt;
  guint total_buffer_cnt;
  struct timeval total_fps_time;
  struct timeval start_fps_time;
  struct timeval last_fps_time;
  struct timeval last_sample_fps_time;
}InstancePerfStruct;

typedef struct
{
  guint num_instances;
  GMutex struct_lock;
  GstPad *sink_bin_pad;
  InstancePerfStruct instance_str[MAX_SOURCE_BINS];
} PerfStructInt;

typedef struct _DsSourceBin
{
    GstElement *source_bin;
    GstElement *uri_decode_bin;
    GstElement *vidconv;
    GstElement *nvvidconv;
    GstElement *capsfilt;
    GstElement *capsraw;
    gint index;
    gboolean is_imagedec;
    gboolean is_streaming;
    guint64 accumulated_base;
    guint64 prev_accumulated_base;
}DsSourceBinStruct;

typedef struct
{
  guint file_loop;
  guint network_type;
  guint num_detected_classes;
  std::string config_path;
  guint seg_gpu_id;
  guint seg_width;
  guint seg_height;
  gboolean seg_background;
  float seg_alpha;
} YamlParasStruct;

static const char* dgpus_unsupport_hw_enc[] = {
  "NVIDIA A100",
  "NVIDIA A30",
  "NVIDIA H100", // NVIDIA H100 SXM, NVIDIA H100 PCIe, NVIDIA H100 NVL
  "NVIDIA T500",
  "GeForce MX570 A",
  "DGX A100"
};

/* Separate a config file entry with delimiters
 * into strings. */
static std::vector<std::string>
split_string (std::string input) {
  std::vector<int> positions;
  for (unsigned int i = 0; i < input.size(); i++) {
    if (input[i] == ';')
      positions.push_back(i);
  }
  std::vector<std::string> ret;
  int prev = 0;
  for (auto &j: positions) {
    std::string temp = input.substr(prev, j - prev);
    ret.push_back(temp);
    prev = j + 1;
  }
  ret.push_back(input.substr(prev, input.size() - prev));
  return ret;
}

static void
parse_tests_yaml (YamlParasStruct *yaml_paras, const gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  std::string paramKey = "";

  for(YAML::const_iterator itr = configyml["tests"].begin();
     itr != configyml["tests"].end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();
    if (paramKey == "file-loop") {
      yaml_paras->file_loop = itr->second.as<guint>();
    }
  }

  for(YAML::const_iterator itr = configyml["primary-gie"].begin();
     itr != configyml["primary-gie"].end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();
    if (paramKey == "config-file-path") {
      yaml_paras->config_path = itr->second.as<std::string>();
    }
  }

  for(YAML::const_iterator itr = configyml["property"].begin();
     itr != configyml["property"].end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();
    if (paramKey == "network-type") {
      yaml_paras->network_type = itr->second.as<guint>();
    }
    if (paramKey == "num-detected-classes") {
      yaml_paras->num_detected_classes = itr->second.as<guint>();
    }
  }
}

static void
parse_filesink_yaml (gint *enc_type, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["filesink"].begin();
     itr != configyml["filesink"].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enc-type") {
      int value = itr->second.as<gint>();
      if(value == 0 || value == 1){
        *enc_type = value;
      }
    } else {
      *enc_type = 0;
    }
  }
  g_print("enc_type:%d\n", *enc_type);
}

static void
parse_segvisual_yaml (YamlParasStruct *yaml_paras, const gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  std::string paramKey = "";

  for(YAML::const_iterator itr = configyml["segvisual"].begin();
     itr != configyml["segvisual"].end(); ++itr)
  {
    paramKey = itr->first.as<std::string>();
    if (paramKey == "gpu-id") {
      yaml_paras->seg_gpu_id = itr->second.as<guint>();
    }

    if (paramKey == "width") {
      yaml_paras->seg_width = itr->second.as<guint>();
    }

    if (paramKey == "height") {
      yaml_paras->seg_height = itr->second.as<guint>();
    }

    if (paramKey == "orig_background") {
      yaml_paras->seg_background = itr->second.as<gboolean>();
    }
    if (paramKey == "alpha") {
      yaml_paras->seg_alpha = itr->second.as<float>();
    }
  }

}

/*
 * Function to seek the source stream to start.
 * It is required to play the stream in loop.
 */
static gboolean
seek_decode (gpointer data)
{
  DsSourceBinStruct *bin = (DsSourceBinStruct *) data;
  gboolean ret = TRUE;

  gst_element_set_state (bin->source_bin, GST_STATE_PAUSED);

  ret = gst_element_seek (bin->source_bin, 1.0, GST_FORMAT_TIME,
      (GstSeekFlags) (GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH),
      GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE);

  if (!ret)
    GST_WARNING ("Error in seeking pipeline");

  gst_element_set_state (bin->source_bin, GST_STATE_PLAYING);

  return FALSE;
}

/**
 * Probe function to drop certain events to support custom
 * logic of looping of each source stream.
 */
static GstPadProbeReturn
restart_stream_buf_prob (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstEvent *event = GST_EVENT (info->data);
  DsSourceBinStruct *bin = (DsSourceBinStruct *) u_data;

  if ((info->type & GST_PAD_PROBE_TYPE_BUFFER)) {
    GST_BUFFER_PTS(GST_BUFFER(info->data)) += bin->prev_accumulated_base;
  }
  if ((info->type & GST_PAD_PROBE_TYPE_EVENT_BOTH)) {
    if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
      g_timeout_add (1, seek_decode, bin);
    }

    if (GST_EVENT_TYPE (event) == GST_EVENT_SEGMENT) {
      GstSegment *segment;

      gst_event_parse_segment (event, (const GstSegment **) &segment);
      segment->base = bin->accumulated_base;
      bin->prev_accumulated_base = bin->accumulated_base;
      bin->accumulated_base += segment->stop;
    }
    switch (GST_EVENT_TYPE (event)) {
      case GST_EVENT_EOS:
        /* QOS events from downstream sink elements cause decoder to drop
         * frames after looping the file since the timestamps reset to 0.
         * We should drop the QOS events since we have custom logic for
         * looping individual sources. */
      case GST_EVENT_QOS:
      case GST_EVENT_SEGMENT:
      case GST_EVENT_FLUSH_START:
      case GST_EVENT_FLUSH_STOP:
        return GST_PAD_PROBE_DROP;
      default:
        break;
    }
  }
  return GST_PAD_PROBE_OK;
}

static void
release_segmentation_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsInferSegmentationMeta *meta = (NvDsInferSegmentationMeta *) user_meta->user_meta_data;
  if (meta->priv_data) {
    gst_mini_object_unref (GST_MINI_OBJECT (meta->priv_data));
  }
  else {
    g_free (meta->class_map);
    g_free (meta->class_probabilities_map);
  }
  delete meta;
}

static gpointer
copy_segmentation_meta (gpointer data, gpointer user_data)
{
  NvDsUserMeta *src_user_meta = (NvDsUserMeta *) data;
  NvDsInferSegmentationMeta *src_meta = (NvDsInferSegmentationMeta *) src_user_meta->user_meta_data;
  NvDsInferSegmentationMeta *meta = (NvDsInferSegmentationMeta *) g_malloc (sizeof (NvDsInferSegmentationMeta));

  meta->classes = src_meta->classes;
  meta->width = src_meta->width;
  meta->height = src_meta->height;
  meta->class_map = (gint *) g_memdup(src_meta->class_map, meta->width * meta->height * sizeof (gint));
  // meta->class_probabilities_map = (gfloat *) g_memdup(src_meta->class_probabilities_map, meta->classes * meta->width * meta->height * sizeof (gfloat));
  meta->class_probabilities_map = NULL;
  meta->priv_data = NULL;

  return meta;
}

static GstPadProbeReturn
pgie_pad_buffer_probe_network_type100 (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;

  GstBuffer *buf = (GstBuffer *)(info->data);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list;
      l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    if (frame_meta && frame_meta->frame_user_meta_list) {
      NvDsFrameMetaList *fmeta_list = NULL;
      NvDsUserMeta *of_user_meta = NULL;

      for (fmeta_list = frame_meta->frame_user_meta_list;
        fmeta_list != NULL; fmeta_list = fmeta_list->next) {
        of_user_meta = (NvDsUserMeta *)fmeta_list->data;
        if (of_user_meta && of_user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
          NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *)(of_user_meta->user_meta_data);
          if (!meta ||
              (meta->num_output_layers != 1) ||
              !(meta->output_layers_info[0].dataType==NvDsInferDataType::INT32 ||
               meta->output_layers_info[0].dataType==NvDsInferDataType::INT64))
            continue;

          NvDsInferLayerInfo *info = &meta->output_layers_info[0];
          info->buffer = meta->out_buf_ptrs_host[0];// cudaMemcpyDeviceToHost is already performed.

          //---Fill NvDsInferSegmentationMeta structure---
          // Acquire a new NvDsUserMeta object from frame_meta.
          NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool (frame_meta->base_meta.batch_meta);
          NvDsInferSegmentationMeta *segmeta = (NvDsInferSegmentationMeta *) g_malloc (sizeof (NvDsInferSegmentationMeta));

          segmeta->classes = numDetectedClasses;
          // Segmentation model ALWAYS has the same W/H as the input tensor.
          segmeta->height = meta->network_info.height;
          segmeta->width = meta->network_info.width;

          printf("seg width %d seg height %d\n", segmeta->width, segmeta->height);

          if(meta->output_layers_info[0].dataType == NvDsInferDataType::INT64) {
            gint *tmp_buf = (gint *) g_malloc (segmeta->width * segmeta->height * sizeof (gint));
            gint *ttmp_buf = tmp_buf;
            for(int i = 0; i < segmeta->width * segmeta->height; i++) {
              *tmp_buf = static_cast<gint> (*(static_cast<gint64 *>(info->buffer)));
              tmp_buf++;
              info->buffer = info->buffer+8; //int64
            }

            segmeta->class_map = (gint *) g_memdup(ttmp_buf, segmeta->width * segmeta->height * sizeof (gint));
            g_free(ttmp_buf);
          } else {
            // Output tensor is the class map already. There is nothing else to parse.
            // Referencing instead of copying info->buffer causes SegV crash.
            segmeta->class_map = (gint *) g_memdup(info->buffer, segmeta->width * segmeta->height * sizeof (gint));
          }
          segmeta->class_probabilities_map = NULL;
          segmeta->priv_data = NULL;

          // Assign NvDsInferSegmentationMeta to the fields of NvDsUserMeta
          user_meta->user_meta_data = segmeta;
          user_meta->base_meta.meta_type = (NvDsMetaType) NVDSINFER_SEGMENTATION_META;
          user_meta->base_meta.release_func = release_segmentation_meta;
          user_meta->base_meta.copy_func = copy_segmentation_meta;

          nvds_add_user_meta_to_frame (frame_meta, user_meta);
          //---Fill NvDsInferSegmentationMeta structure---
        }
      }
    }
  }

  // use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

static gboolean
perf_measurement_callback (gpointer data)
{
  PerfStructInt *str = (PerfStructInt *) data;
  guint buffer_cnt[MAX_SOURCE_BINS];
  PerfStruct perf_struct;
  struct timeval current_fps_time;
  guint i;
  static guint header_print_cnt = 0;
  if (header_print_cnt % 20 == 0) {
    g_print ("\n**PERF:  ");
    for (i = 0; i < str->num_instances; i++) {
      g_print ("FPS %d (Avg)\t", i);
    }
    g_print ("\n");
    header_print_cnt = 0;
  }
  header_print_cnt++;

  time_t t = time (NULL);
  struct tm *tm = localtime (&t);
  g_print ("%s", asctime (tm));

  g_mutex_lock (&str->struct_lock);

  for (i = 0; i < str->num_instances; i++) {
    buffer_cnt[i] =
        str->instance_str[i].buffer_cnt;
    str->instance_str[i].buffer_cnt = 0;
  }

  perf_struct.num_instances = str->num_instances;
  gettimeofday (&current_fps_time, NULL);

  g_print ("**PERF:  ");
  for (i = 0; i < str->num_instances; i++) {
    InstancePerfStruct *str1 = &str->instance_str[i];
    gdouble time1 =
        (str1->total_fps_time.tv_sec +
        str1->total_fps_time.tv_usec / 1000000.0) +
        (current_fps_time.tv_sec + current_fps_time.tv_usec / 1000000.0) -
        (str1->start_fps_time.tv_sec +
        str1->start_fps_time.tv_usec / 1000000.0);

    gdouble time2;

    if (str1->last_sample_fps_time.tv_sec == 0 &&
        str1->last_sample_fps_time.tv_usec == 0) {
      time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->start_fps_time.tv_sec +
          str1->start_fps_time.tv_usec / 1000000.0);
    } else {
      time2 =
          (str1->last_fps_time.tv_sec +
          str1->last_fps_time.tv_usec / 1000000.0) -
          (str1->last_sample_fps_time.tv_sec +
          str1->last_sample_fps_time.tv_usec / 1000000.0);
    }
    str1->total_buffer_cnt += buffer_cnt[i];
    perf_struct.fps[i] = buffer_cnt[i] / time2;
    if (isnan (perf_struct.fps[i]))
      perf_struct.fps[i] = 0;

    perf_struct.fps_avg[i] = str1->total_buffer_cnt / time1;
    if (isnan (perf_struct.fps_avg[i]))
      perf_struct.fps_avg[i] = 0;

    str1->last_sample_fps_time = str1->last_fps_time;

    g_print ("%.2f(%.2f)\t", perf_struct.fps[i], perf_struct.fps_avg[i]);
  }

  g_print("\n");

  g_mutex_unlock (&str->struct_lock);

  return TRUE;
}

/**
 * Buffer probe function on element.
 */
static GstPadProbeReturn
buf_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  PerfStructInt *str = (PerfStructInt *) u_data;
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  if (!batch_meta)
    return GST_PAD_PROBE_OK;

  g_mutex_lock (&str->struct_lock);
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    InstancePerfStruct *str1 = &str->instance_str[frame_meta->pad_index];
    gettimeofday (&str1->last_fps_time, NULL);
    if (str1->start_fps_time.tv_sec == 0 && str1->start_fps_time.tv_usec == 0) {
      str1->start_fps_time = str1->last_fps_time;
    } else {
      str1->buffer_cnt++;
    }
  }
  g_mutex_unlock (&str->struct_lock);
  return GST_PAD_PROBE_OK;
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
        g_printerr ("Failed to link decoderbin src pad to converter sink pad\n");
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
        g_printerr ("Failed to link decoderbin src pad to converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
    }
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
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
  g_print ("Decodebin child added: %s\n", name);
  DsSourceBinStruct *src_bin_st = (DsSourceBinStruct *) user_data;
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }

  if (g_strstr_len (name, -1, "nvv4l2decoder") == name) {
    if (fileLoop && !src_bin_st->is_streaming) {
      g_print ("loop model: %s\n", name);
      GstPad *gstpad = gst_element_get_static_pad (GST_ELEMENT(object), "sink");
      gst_pad_add_probe(gstpad, (GstPadProbeType) (GST_PAD_PROBE_TYPE_EVENT_BOTH |
              GST_PAD_PROBE_TYPE_EVENT_FLUSH | GST_PAD_PROBE_TYPE_BUFFER), restart_stream_buf_prob, user_data, NULL);
      gst_object_unref (gstpad);
    }
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

  if (!ds_source_struct->source_bin || !ds_source_struct->uri_decode_bin
      || !ds_source_struct->nvvidconv
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
      gst_ghost_pad_new("src", gstpad))) {
    g_printerr ("Could not add ghost pad in '%s'",
        GST_ELEMENT_NAME(ds_source_struct->capsfilt));
  }
  gst_object_unref (gstpad);

  return true;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
        g_print ("End of stream\n");
        g_main_loop_quit (loop);
        break;
    case GST_MESSAGE_ERROR: {
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

static bool
is_enc_hw_support() {
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  bool enc_hw_support = TRUE;
  if (prop.integrated) {
    char device_name[50];
    FILE* ptr = fopen("/proc/device-tree/model", "r");

    if (ptr) {
      while (fgets(device_name, 50, ptr) != NULL) {
        if (strstr(device_name,"Orin") && (strstr(device_name,"Nano")))
          enc_hw_support = FALSE;
        }
    }
    fclose(ptr);
  } else {
    for (int i = 0; i < sizeof(dgpus_unsupport_hw_enc)/sizeof(dgpus_unsupport_hw_enc[0]); i++) {
      if (!strncasecmp(prop.name, dgpus_unsupport_hw_enc[i], strlen(dgpus_unsupport_hw_enc[i]))) {
        enc_hw_support = FALSE;
        break;
      }
    }
  }
  return enc_hw_support;
}

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

static void printUsage(const char* cmd) {
    g_printerr ("\tUsage: %s -c pgie_config_file -i <H264 or JPEG filename> [-b BATCH]"
      " [-d]\n\tOR\n\t %s yml_config_file\n", cmd, cmd);
    g_printerr ("-h: \n\tprint help info \n");
    g_printerr ("-c: \n\tpgie config file, e.g. pgie_frcnn_tao_config.txt  \n");
    g_printerr ("-i: \n\tH264 or JPEG input file  \n");
    g_printerr ("-b: \n\tbatch size, this will override the value of \"batch-size\" in pgie config file  \n");
    g_printerr ("-d: \n\tenable display, otherwise dump to output H264 or JPEG file  \n");
    g_printerr ("-f: \n\tuse fake_sink to test the performace\n");
    g_printerr ("-l: \n\tloop mode for the pipeline\n");
    g_printerr ("-o: \n\tOriginal background On\n");
    g_printerr ("-a: \n\tAlpha value with original background setting\n");
    g_printerr ("-w: \n\tThe model output width\n");
    g_printerr ("-e: \n\tThe model output height\n");
    g_printerr ("yml_config_file: \n\tYAML config file, e.g. seg_app_unet.yml \n");
}
int
main (int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source_bin = NULL,
               *streammux = NULL, *sink = NULL,
               *pgie = NULL, *nvvidconv = NULL, *segvisual = NULL,
               *parser1 = NULL, *nvvidconv1 = NULL, *enc = NULL,
               *tiler = NULL, *mux = NULL;

    DsSourceBinStruct source_struct[128];

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    GstBus *bus = NULL;
    guint bus_watch_id;

    gboolean isImage = FALSE;
    gboolean useDisplay = FALSE;
    gboolean useFakeSink = FALSE;
    gboolean original_background = FALSE;
    float alpha=1.0f;
    guint tiler_rows, tiler_cols;
    guint batchSize = 0;
    guint pgie_batch_size;
    guint c;
    const char* optStr = "a:b:c:w:e:dohfli:";
    std::string pgie_config;
    gboolean isYAML = FALSE;
    GList* g_list = NULL;
    GList* iterator = NULL;
    static guint src_cnt = 0;
    PerfStructInt str;
    YamlParasStruct yaml_paras;
    fileLoop = 0;
    int enc_type = 0;
    networkType = 2;
    numDetectedClasses = 1;
    guint model_width = 0;
    guint model_height = 0;
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    if (argc==2 && (g_str_has_suffix(argv[1], ".yml") ||
        g_str_has_suffix(argv[1], ".yaml"))) {
        isYAML = TRUE;
        if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
          g_printerr ("No source is found. Exiting.\n");
          return -1;
        }

        RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1],
                "primary-gie"));
        if(pgie_type == NVDS_GIE_PLUGIN_INFER){
          parse_tests_yaml(&yaml_paras, argv[1]);
          fileLoop = yaml_paras.file_loop;
          yaml_paras.config_path.erase(0,2);
          yaml_paras.config_path = "configs" + yaml_paras.config_path;
          parse_tests_yaml(&yaml_paras, yaml_paras.config_path.c_str());
          networkType = yaml_paras.network_type;
          numDetectedClasses = yaml_paras.num_detected_classes;
        }
        parse_tests_yaml(&yaml_paras, argv[1]);

        fileLoop = yaml_paras.file_loop;
        yaml_paras.config_path.erase(0,2);
        yaml_paras.config_path = "configs" + yaml_paras.config_path;
        networkType = yaml_paras.network_type;
        numDetectedClasses = yaml_paras.num_detected_classes;
    } else {
        while ((c = getopt(argc, argv, optStr)) != -1) {
            switch (c) {
                case 'a':
                    alpha = std::atof(optarg);
                    break;
                case 'b':
                    batchSize = std::atoi(optarg);
                    batchSize = batchSize == 0 ? 1:batchSize;
                    break;
                case 'c':
                {
                    pgie_config.assign(optarg);
                    GKeyFile *key_file = g_key_file_new ();
                    GError *error = NULL;
                    g_key_file_load_from_file (key_file, pgie_config.c_str(), G_KEY_FILE_NONE, &error);
                    gboolean has_key;
                    has_key = g_key_file_has_key(key_file, "property", "network-type", &error);
                    if (has_key) {
                      networkType = g_key_file_get_integer (key_file, "property", "network-type", &error);
                    }
                    has_key = g_key_file_has_key(key_file, "property", "num-detected-classes", &error);
                    if (has_key) {
                      numDetectedClasses = g_key_file_get_integer (key_file, "property", "num-detected-classes", &error);
                    }
                    g_key_file_free(key_file);
                }
                    break;
                case 'd':
                    useDisplay = TRUE;
                    break;
                case 'f':
                    useFakeSink = TRUE;
                    break;
                case 'i':
                    g_list = g_list_append(g_list, optarg);
                    break;
                case 'l':
                    fileLoop = 1;
                    break;
                case 'o':
                    original_background = TRUE;
                    break;
                case 'w':
                    model_width = std::atoi(optarg);
                    break;
                case 'e':
                    model_height = std::atoi(optarg);
                    break;
                case 'h':
                default:
                    printUsage(argv[0]);
                    return -1;
            }
        }
    }

    /* Check input arguments */
    if (argc == 1) {
        printUsage(argv[0]);
        return -1;
    }

    if(useDisplay) useFakeSink = FALSE;
    
    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("ds-custom-pipeline");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    gst_bin_add (GST_BIN (pipeline), streammux);

    for (iterator = g_list, src_cnt=0; iterator; iterator = iterator->next,src_cnt++) {
      /* Source element for reading from the file */
      source_struct[src_cnt].index = src_cnt;

      if (g_strrstr ((gchar *)iterator->data, ".jpg") || g_strrstr ((gchar *)iterator->data, ".jpeg")
          || g_strrstr ((gchar *)iterator->data, ".png"))
        isImage = TRUE;
      else
        isImage = FALSE;
      if (g_strrstr ((gchar *)iterator->data, "rtsp://") || g_strrstr ((gchar *)iterator->data, "v4l2://")
          || g_strrstr ((gchar *)iterator->data, "http://") || g_strrstr ((gchar *)iterator->data, "rtmp://")) {
        source_struct[src_cnt].is_streaming = TRUE;
      } else {
        source_struct[src_cnt].is_streaming = FALSE;
      }
      if (!create_source_bin (&(source_struct[src_cnt]), (gchar *)iterator->data))
      {
        g_printerr ("Source bin could not be created. Exiting.\n");
        return -1;
      }

      gst_bin_add (GST_BIN (pipeline), source_struct[src_cnt].source_bin);

      g_snprintf (pad_name_sink, 64, "sink_%d", src_cnt);
      sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
      g_print("Request %s pad from streammux\n",pad_name_sink);
      if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
      }

      srcpad = gst_element_get_static_pad (source_struct[src_cnt].source_bin,
          pad_name_src);
      if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
      }

      if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
      }
      gst_object_unref (sinkpad);
      gst_object_unref (srcpad);

    }

    str.num_instances = src_cnt;
    g_timeout_add (5000, perf_measurement_callback, &str);

    const char* batch_size = std::getenv("BATCH_SIZE");
    if(batch_size != NULL ) {
        batchSize = std::stoi(batch_size);
        g_printerr("batch size is %d \n", batchSize);
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      pgie = gst_element_factory_make ("nvinferserver", "primary-nvinference-engine");
    } else {
      pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    }

    /* Use convertor to convert from NV12 to RGBA as required by segvisual */
    //nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    if(!model_width)
      model_width = SEG_OUTPUT_WIDTH;
    if(!model_height)
      model_height = SEG_OUTPUT_HEIGHT;
    if(isYAML) {
        parse_segvisual_yaml(&yaml_paras, argv[1]);
        if (!yaml_paras.seg_width || !yaml_paras.seg_height) {
            g_printerr ("segvisual resolution should not be zero. Exiting.\n");
            return -1;
        }
        model_height = yaml_paras.seg_height;
        model_width = yaml_paras.seg_width;
        original_background = yaml_paras.seg_background;
        alpha = yaml_paras.seg_alpha;
    }
    segvisual = gst_element_factory_make ("nvsegvisual", "nv-segvisual");
    g_object_set (G_OBJECT (segvisual), "original-background", original_background, NULL);
    g_object_set (G_OBJECT (segvisual), "alpha", alpha, NULL);

    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    if(isYAML) {
        GstElement *eglsink = gst_element_factory_make ("nveglglessink", "test-egl-sink");
        GstElement *filesink = gst_element_factory_make ("filesink", "test-file-sink");
        GstElement *fakesink = gst_element_factory_make("fakesink", "test-fake-sink");
        if(NVDS_YAML_PARSER_DISABLED != nvds_parse_egl_sink(eglsink, argv[1], "eglsink")){
            useDisplay = TRUE;
        } else if (NVDS_YAML_PARSER_DISABLED != nvds_parse_file_sink(filesink, argv[1], "filesink")){
            useDisplay = FALSE;
        } else if (NVDS_YAML_PARSER_DISABLED != nvds_parse_egl_sink(eglsink, argv[1], "fakesink")){
            useDisplay = FALSE;
            useFakeSink = TRUE;
        } else {
            g_printerr ("No sink is configured. Exiting.\n");
            return -1;
        }
        g_object_unref(eglsink);
        g_object_unref(filesink);
        g_object_unref(fakesink);
    }

    const char* use_display = std::getenv("USE_DISPLAY");
    if(use_display != NULL && std::stoi(use_display) == 1) {
        useDisplay = true;
    }

    if(useDisplay == FALSE) {
        if(isImage == FALSE){
            parser1 = gst_element_factory_make ("h264parse", "h264-parser1");
            if (isYAML) {
              parse_filesink_yaml(&enc_type, argv[1]);
            } else {
              // 0: HW 1: SW
              enc_type = is_enc_hw_support() ? 0 : 1;
            }
            if(enc_type == 0){
              enc = gst_element_factory_make ("nvv4l2h264enc", "h264-enc");
            } else {
              enc = gst_element_factory_make ("x264enc", "h264-enc");
            }
            if(!useFakeSink) {
                mux = gst_element_factory_make ("qtmux", "mp4-mux");
                if (!mux) {
                    g_printerr ("Failed to create mp4-mux");
                    return -1;
                }
                gst_bin_add (GST_BIN (pipeline), mux);
            }
        } else {
            parser1 = gst_element_factory_make ("jpegparse", "jpeg-parser1");
            enc = gst_element_factory_make ("jpegenc", "jpeg-enc");
        }
        nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
        if(!useFakeSink) {
            sink = gst_element_factory_make ("filesink", "file-sink");
        } else {
            sink = gst_element_factory_make("fakesink", "file-sink");
        }
        if (!pgie
                || !tiler || !nvvidconv1 || !segvisual || !enc || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }

        //save the file to local dir
        if(isImage == FALSE)
            g_object_set (G_OBJECT (sink), "location", "./out.mp4", NULL);
        else
            g_object_set (G_OBJECT (sink), "location", "./out.jpg", NULL);
    } else {
        if(prop.integrated)
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        else
#ifdef __aarch64__
            sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
#else
            sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
        if (!pgie
                || !tiler || !segvisual || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    }

    if(isYAML) {
        nvds_parse_streammux(streammux, argv[1], "streammux");
        if(!batchSize) {
            g_object_get(G_OBJECT (streammux), "batch-size", &batchSize, NULL);
        }
    }

    if(!batchSize) { batchSize = src_cnt; }

    g_print ("batchSize %d...\n", batchSize);

    if(source_struct[0].is_streaming == TRUE)
        g_object_set (G_OBJECT (streammux), "live-source", true, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
              MUXER_OUTPUT_HEIGHT, "batch-size", batchSize,
              "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    g_object_set (G_OBJECT (segvisual), "width", model_width, "height",
              model_height, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    if(isYAML) {
        nvds_parse_gie (pgie, argv[1], "primary-gie");
    } else {
        g_object_set (G_OBJECT (pgie),
                  "config-file-path", pgie_config.c_str(), NULL);
    }

    /* Override the batch-size set in the config file with the number of sources. */
    g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
    if (pgie_batch_size != batchSize) {
        g_printerr
          ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
          pgie_batch_size, batchSize);
        g_object_set (G_OBJECT (pgie), "batch-size", batchSize, NULL);
    }

    tiler_rows = (guint) sqrt (batchSize);
    tiler_cols = (guint) ceil (1.0 * batchSize / tiler_rows);
    /* we set the tiler properties here */
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_cols,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);
    g_object_set (G_OBJECT (segvisual), "batch-size", batchSize, NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    if(useDisplay == FALSE) {
        gst_bin_add_many (GST_BIN (pipeline), pgie, tiler,
                           segvisual, nvvidconv1, enc, parser1, sink, NULL);
    } else {
        gst_bin_add_many (GST_BIN (pipeline), pgie,
                          tiler, segvisual, sink, NULL);
    }

    /* We link the elements together */
    /* uridocoderbin ->
     * nvinfer -> nvvideoconvert -> segvisual -> video-renderer */
    if (useDisplay == FALSE) {
        if (isImage == FALSE && !useFakeSink) {
            if (!gst_element_link_many (streammux, pgie,
                                         segvisual, tiler, nvvidconv1, enc, parser1, mux, sink, NULL)) {
                g_printerr ("Elements could not be linked: 2. Exiting.\n");
                return -1;
            }
        } else {
            if (!gst_element_link_many (streammux, pgie,
                                         segvisual, tiler, nvvidconv1, enc, parser1, sink, NULL)) {
                g_printerr ("Elements could not be linked: 2. Exiting.\n");
                return -1;
            }
        }
    } else {
        if (!gst_element_link_many (streammux, pgie,
                                     tiler, segvisual, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    }

    /*Performance measurement video fps*/
    GstPad *streammux_src_pad = gst_element_get_static_pad (streammux, "src");
    if (!streammux_src_pad)
      g_print ("Unable to get streammux src pad\n");
    else
      gst_pad_add_probe(streammux_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
          buf_probe, &str, NULL);
    gst_object_unref (streammux_src_pad);

    /*post-process for network-type = 100,
     *nvinferser native postprocess can't process int output datatype.*/
    if (networkType == 100 || pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
        GstPad *pgie_src_pad = gst_element_get_static_pad (pgie, "src");
        if (!pgie_src_pad)
          g_print ("Unable to get streammux src pad\n");
        else
          gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
              pgie_pad_buffer_probe_network_type100, NULL, NULL);
        gst_object_unref (pgie_src_pad);
    }

    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", pgie_config.c_str());
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
