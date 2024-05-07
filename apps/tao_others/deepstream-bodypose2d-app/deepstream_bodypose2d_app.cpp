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

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <locale.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gmodule.h>
#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"
#include "nvds_version.h"
#include "nvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsinfer.h"
#include "cuda_runtime_api.h"
#include "ds_bodypose2d_meta.h"
#include "cv/core/Tensor.h"
#include "nvbufsurface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <gst/rtsp-server/rtsp-server.h>
#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"
#include <yaml-cpp/yaml.h>

using namespace std;
using std::string;

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_PERSON 1

#define PGIE_DETECTED_CLASS_NUM 4

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 640
#define MUXER_OUTPUT_HEIGHT 480

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CONFIG_GPU_ID "gpu-id"

gint frame_number = 0;
gint total_person_num = 0;
/*
  For the pre-trained default model
   0 - nose
   1 - neck
   2 - right shoulder
   3 - right elbow
   4 - right hand
   5 - left shoulder
   6 - left elbow
   7 - left hand
   8 - right hip
   9 - right knee
   10 - right foot
   11 - left hip
   12 - left knee
   13 - left foot
   14 - right eye
   15 - left eye
   16 - right ear 
   17 - left ear*/

std::vector<std::pair<uint32_t, uint32_t>> display_connect_table;

#define PRIMARY_DETECTOR_UID 1

typedef struct _perf_measure{
  GstClockTime pre_time;
  GstClockTime total_time;
  guint count;
}perf_measure;

typedef struct _DsSourceBin
{
    GstElement *source_bin;
    GstElement *uri_decode_bin;
    GstElement *vidconv;
    GstElement *nvvidconv;
    GstElement *capsfilt;
    GstElement *capsraw;
    gint index;
    bool is_imagedec;
}DsSourceBinStruct;

GstElement *pipeline = NULL;

static void
signal_catch_callback(int signum)
{
  g_print("User Interrupted..\n");
  if(pipeline != NULL) {
    gst_element_send_event(pipeline, gst_event_new_eos());
    g_print("Send EOS to pipline!\n");
  }
}

static void
signal_catch_setup()
{
  struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_handler = signal_catch_callback;
  sigaction(SIGINT, &action, NULL);
}

/* Calculate performance data, draw bodypart lines to display */
/* the bodypose2d results                                     */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsObjectMeta *obj_meta = NULL;
  guint person_count = 0;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  GstClockTime now;
  perf_measure * perf = (perf_measure *)(u_data);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  now = g_get_monotonic_time();

  if (perf->pre_time == GST_CLOCK_TIME_NONE) {
    perf->pre_time = now;
    perf->total_time = GST_CLOCK_TIME_NONE;
  } else {
    if (perf->total_time == GST_CLOCK_TIME_NONE) {
      perf->total_time = (now - perf->pre_time);
    } else {
      perf->total_time += (now - perf->pre_time);
    }
    perf->pre_time = now;
    perf->count++;
  }

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
 
    if (!frame_meta)
      continue;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      if (!obj_meta)
        continue;
        
      /* Calculate the person number. */
      if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
        if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
          person_count++;
      }
    }
  }

  g_print ("Frame Number = %d Person Count = %d\n",
           frame_number, person_count);
  frame_number++;
  total_person_num += person_count;
  return GST_PAD_PROBE_OK;
}

/*Generate bodypose2d display meta right after inference */
static GstPadProbeReturn
tile_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList * l_frame = NULL;
  NvDsMetaList * l_obj = NULL;
  int part_index = 0;
  int body_connect_num = display_connect_table.size();

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    NvDsDisplayMeta *disp_meta = NULL;
 
    if (!frame_meta)
      continue;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      if (!obj_meta)
        continue;
        
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if(user_meta->base_meta.meta_type ==
            (NvDsMetaType) NVDS_USER_RIVA_META_2DPOSE) {
          NvDs2DposeMetaData *body2d_meta =
              (NvDs2DposeMetaData *) user_meta->user_meta_data;
          std::map<int, BodyPartLoc *> bodypart_map;
          int line_num = 0;
          /* Generate the coordinate map with the reasonable detected parts,*/
          /* e.g. the part's score is higher than 0.0                       */
          for (part_index = 0;part_index < body2d_meta->bodypart_num;
              part_index++) {
            if (body2d_meta->bodypart_locs[part_index].score > 0.0)
                bodypart_map[body2d_meta->bodypart_locs[part_index].part_idx] =
                  &(body2d_meta->bodypart_locs[part_index]);
          }
          /*If the parts pair can be found in coordinates map, the connection*/
          /*can be displayed, so the pair will be added to display meta as */
          /*a line*/
          for (line_num = 0; line_num < body_connect_num; line_num++) {
            if ((bodypart_map.count(display_connect_table[line_num].first))
                && (bodypart_map.count(display_connect_table[line_num].second))) {
              if (!disp_meta) {
                disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                disp_meta->num_lines = 0;
              } else {
                if (disp_meta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) {
                  nvds_add_display_meta_to_frame (frame_meta, disp_meta);
                  disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                  disp_meta->num_lines = 0;
                }
              }
              disp_meta->line_params[disp_meta->num_lines].x1 =
                  bodypart_map[display_connect_table[line_num].first]->x;
              disp_meta->line_params[disp_meta->num_lines].y1 =
                  bodypart_map[display_connect_table[line_num].first]->y;
              disp_meta->line_params[disp_meta->num_lines].x2 =
                  bodypart_map[display_connect_table[line_num].second]->x;
              disp_meta->line_params[disp_meta->num_lines].y2 =
                  bodypart_map[display_connect_table[line_num].second]->y;
              disp_meta->line_params[disp_meta->num_lines].line_width = 3;
              disp_meta->line_params[disp_meta->num_lines].line_color.red = 0.0;
              disp_meta->line_params[disp_meta->num_lines].line_color.green = 1.0;
              disp_meta->line_params[disp_meta->num_lines].line_color.blue = 1.0;
              disp_meta->line_params[disp_meta->num_lines].line_color.alpha = 0.5;
              disp_meta->num_lines++;
            }
          }        
        }
      }
    }
    if(disp_meta)
      nvds_add_display_meta_to_frame (frame_meta, disp_meta);
  }

  return GST_PAD_PROBE_OK;
}

std::unique_ptr< cvcore::bodypose2d::BodyPose2DPostProcessor > body2Dposepost;

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes and othet
 * model output. The result would be attached as object-meta(NvDsObjectMeta)
 * into the same frame metadata.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  int frame_width, frame_height;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (GST_BUFFER (info->data), &in_map_info, GST_MAP_READ)) {
    g_printerr ("Failed to map GstBuffer\n");
    return GST_PAD_PROBE_PASS;
  }
  in_surf = (NvBufSurface *) in_map_info.data;
  gst_buffer_unmap(GST_BUFFER (info->data), &in_map_info);

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    frame_width = in_surf->surfaceList[frame_meta->batch_id].width;
    frame_height = in_surf->surfaceList[frame_meta->batch_id].height;

    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

      NvDsInferTensorMeta *meta =
          (NvDsInferTensorMeta *) user_meta->user_meta_data;
      std::size_t heatmap_w;
      std::size_t heatmap_h;
      std::size_t heatmap_c;
      std::size_t pafmap_w;
      std::size_t pafmap_h;
      std::size_t pafmap_c;
      int post_w;
      int post_h;
      float top_offset, left_offset;
      float * heatmap_data;
      float * pafmap_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];

        NvDsInferDimsCHW LayerDims;
        std::vector < NvDsInferLayerInfo >
          outputLayersInfo (meta->output_layers_info,
          meta->output_layers_info + meta->num_output_layers);
        getDimsCHWFromDims(LayerDims, outputLayersInfo[i].inferDims);
        //Prepare CVCORE input layers
        if (strcmp(outputLayersInfo[i].layerName,
            "conv2d_transpose_1/BiasAdd:0") == 0) {
          //The layer of pafmap contains 112x160x38 float
          pafmap_data = (float *)meta->out_buf_ptrs_host[i];
          pafmap_w = LayerDims.w;
          pafmap_h = LayerDims.h;
          pafmap_c = LayerDims.c;
        } else if (strcmp(outputLayersInfo[i].layerName,
            "heatmap_out/BiasAdd:0") == 0) {
          //The layer of heatmap contains 28x40x19 float
          heatmap_data = (float *)meta->out_buf_ptrs_host[i];
          heatmap_w = LayerDims.w;
          heatmap_h = LayerDims.h;
          heatmap_c = LayerDims.c;
        }
      }

      cvcore::Tensor<cvcore::NCHW, cvcore::CX, cvcore::F32> tempPafmap(
          pafmap_w, pafmap_h, 1, pafmap_c, (float *)pafmap_data, true);
      cvcore::Tensor<cvcore::NCHW, cvcore::CX, cvcore::F32> tempHeatmap(
          heatmap_w, heatmap_h, 1, heatmap_c, (float *)heatmap_data, true);

      //Prepare output array
      cvcore::Array<cvcore::ArrayN< cvcore::bodypose2d::Human,
          cvcore::bodypose2d::BodyPose2D::MAX_HUMAN_COUNT >> output(1, true);
      output.setSize(1);
      
      if (meta->network_info.width * frame_height <
          meta->network_info.height * frame_width) {
          post_w = frame_width;
          post_h = (frame_width * meta->network_info.height +
              meta->network_info.width/2)/meta->network_info.width;
          top_offset = (meta->network_info.height * frame_width)/(float)meta->network_info.width/2.0 - frame_height/2.0;
          left_offset = 0.0;
      } else {
          post_w = (frame_height * meta->network_info.width +
              meta->network_info.height/2)/meta->network_info.height;
          post_h = frame_height;
          left_offset = (meta->network_info.width * frame_height)/(float)meta->network_info.height/2.0 - frame_width/2.0;
          top_offset = 0.0;
      }

      body2Dposepost->execute(output, tempPafmap, tempHeatmap,
          post_w, post_h);
      
      int human_num = output[0].getSize();

      for (int i = 0; i < human_num; i++)
      {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
        cvcore::bodypose2d::Human human;

        human =  output[0][i];

        obj_meta->unique_component_id = meta->unique_id;
        obj_meta->confidence = 1.0;
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = PGIE_CLASS_ID_PERSON;

        NvOSD_RectParams & rect_params = obj_meta->rect_params;

        /* Assign bounding box coordinates. */
        rect_params.left = human.boundingBox.xmin - left_offset;
        rect_params.top = human.boundingBox.ymin - top_offset;
        rect_params.width = human.boundingBox.xmax - human.boundingBox.xmin;
        rect_params.height = human.boundingBox.ymax - human.boundingBox.ymin;
        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

        /*add user meta for 2dpose*/
        if (!nvds_add_2dpose_meta (batch_meta, obj_meta, human, frame_width,
          frame_height, left_offset, top_offset)) {
          g_printerr ("Failed to get bbox from model output\n");
        }

        nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

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
    gst_bin_add (GST_BIN (bin_struct->source_bin), bin_struct->vidconv);
    bin_struct->is_imagedec=true;
  } else {
    bin_struct->vidconv = NULL;
    bin_struct->is_imagedec=false;
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

std::vector<std::string> split(std::string str, std::string pattern)
{
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern;
    int size = str.size();
    for (int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            s.erase(0,s.find_first_not_of(" "));
            s.erase(s.find_last_not_of(" ") + 1);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *streammux = NULL, *sink = NULL,
             *primary_detector = NULL, *nvtile = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL,
             *rtppay = NULL, *mux = NULL, *encparse = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL;
  DsSourceBinStruct source_struct[128];
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstCaps *caps = NULL;
  //int i;
  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;
  perf_measure perf_measure;

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";
  gchar *filepath;
  ifstream fconfig;
  std::map<string, float> postprocess_params_list;
  bool isYAML=false;
  bool isImage=false;
  bool isStreaming=false;
  GList* g_list = NULL;
  GList* iterator = NULL;
  bool isH264 = true;
  int enc_type = ENCODER_TYPE_HW;

  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
    
  /* Check input arguments */
  if (argc == 2 && (g_str_has_suffix(argv[1], ".yml")
      || (g_str_has_suffix(argv[1], ".yaml")))) {
    isYAML=TRUE;
    if(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie") == NVDS_YAML_PARSER_SUCCESS) {
      g_print("pgie_type %d\n", pgie_type);
    }
  } else {
    if (argc < 7 || argc > 134 || (atoi(argv[1]) != 1 && atoi(argv[1]) != 2 &&
        atoi(argv[1]) != 3 && atoi(argv[1]) != 4)) {
      g_printerr ("Usage: %s [1:file sink|2:fakesink|3:display sink|4:rtsp output] "
        "<model configure file> <udp port> <rtsp port> "
        "<input uri> ... <input uri> <out H264 filename>\n"
        "OR\n%s <yaml config file>\n", argv[0], argv[0]);
      return -1;
    }
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  /* setup singal handler */
  signal_catch_setup();
  loop = g_main_loop_new (NULL, FALSE);
  
  perf_measure.pre_time = GST_CLOCK_TIME_NONE;
  perf_measure.total_time = GST_CLOCK_TIME_NONE;
  perf_measure.count = 0;  

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("pipeline"); 

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
      g_printerr ("One main element could not be created. Exiting.\n");
      return -1;
  }

  gst_bin_add (GST_BIN(pipeline), streammux);

 
  /* Multiple source files */
  if(!isYAML) {
    for (src_cnt=0; src_cnt<(guint)argc-6; src_cnt++) {
      g_list = g_list_append(g_list, argv[src_cnt + 5]);
    }
  } else {
      if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
        g_printerr ("No source is found. Exiting.\n");
        return -1;
      }
  }

  for (iterator = g_list, src_cnt=0; iterator; iterator = iterator->next,src_cnt++) {
    /* Source element for reading from the file */
    source_struct[src_cnt].index = src_cnt;

    if (g_strrstr ((gchar *)iterator->data, ".jpg") || g_strrstr ((gchar *)iterator->data, ".jpeg")
        || g_strrstr ((gchar *)iterator->data, ".png"))
      isImage = true;
    else
      isImage = false;
    if (g_strrstr ((gchar *)iterator->data, "rtsp://") || g_strrstr ((gchar *)iterator->data, "v4l2://")
        || g_strrstr ((gchar *)iterator->data, "http://") || g_strrstr ((gchar *)iterator->data, "rtmp://")) {
      isStreaming = true;
    } else {
      isStreaming = false;
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

  /* Create three nvinfer instances for bodypose2d*/
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    primary_detector = gst_element_factory_make ("nvinferserver",
                         "primary-infer-engine1");
  } else {
    primary_detector = gst_element_factory_make ("nvinfer",
                         "primary-infer-engine1");
  }

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvid-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvid-converter1");

  capfilt = gst_element_factory_make ("capsfilter", "nvvideo-caps");

  nvtile = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

 
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");
  queue6 = gst_element_factory_make ("queue", "queue6");

  guint output_type = 0;
  if (isYAML) {
    output_type = ds_parse_group_type(argv[1], "output");
    if(!output_type){
      g_printerr ("No output setting. Exiting.\n");
      return -1;
    }
  } else {
    output_type = atoi(argv[1]);
  }

  if (output_type == 1) {
    GString * filename = NULL;
    if (isYAML)
        filename = ds_parse_file_name(argv[1], "output");
    else
        filename = g_string_new(argv[argc-1]);

    if (isImage) {
      outenc = gst_element_factory_make ("jpegenc", "jpegenc");
      caps =
          gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
          "I420", NULL);
      g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
      filepath = g_strconcat(filename->str,".jpg",NULL);
    } else {
      mux = gst_element_factory_make ("qtmux", "mp4-mux");
      if(isYAML) {
        isH264 = !(ds_parse_enc_codec(argv[1], "output"));
        enc_type = ds_parse_enc_type(argv[1], "output");
      }

      create_video_encoder(isH264, enc_type, &capfilt, &outenc, &encparse, NULL);
      filepath = g_strconcat(filename->str,".mp4",NULL);
      if (isYAML) {
        if(enc_type == ENCODER_TYPE_HW)
          ds_parse_enc_config (outenc, argv[1], "output");
      } else {
        g_object_set (G_OBJECT (outenc), "bitrate", 4000000, NULL);
      }
    }
    g_string_free(filename, TRUE);
    sink = gst_element_factory_make ("filesink", "nvvideo-renderer");
  }
  else if (output_type == 2)
    sink = gst_element_factory_make ("fakesink", "fake-renderer");
  else if (output_type == 3) {
    if(prop.integrated)
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
    else
#ifdef __aarch64__
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
#else
      sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }
  else if (output_type == 4) {
    if (isImage) {
         g_printerr ("Can not support RTSP output for image. Exiting.\n");
         return -1;
    }
    sink = gst_element_factory_make ("udpsink", "nv-udpsink");
  }

  if (!primary_detector || !nvvidconv
      || !nvosd || !sink  || !capfilt ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (isYAML)
    nvds_parse_streammux(streammux, argv[1], "streammux");
  else
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC,
      NULL);
  if (isStreaming)
    g_object_set (G_OBJECT (streammux), "live-source", true, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", src_cnt, NULL);

  tiler_rows = (guint) sqrt (src_cnt);
  tiler_columns = (guint) ceil (1.0 * src_cnt / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", 1280, "height", 720, NULL);

  if(isYAML)
    nvds_parse_gie (primary_detector, argv[1], "primary-gie");
  else
    g_object_set (G_OBJECT (primary_detector), "config-file-path",
      "../../../configs/nvinfer/bodypose2d_tao/bodypose2d_pgie_config.txt",
      "unique-id", PRIMARY_DETECTOR_UID, NULL);

  /* we add a bus message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), primary_detector, queue1, queue2, 
      queue3, queue4, nvvidconv, nvosd, nvtile, sink,
      NULL);

  if (!gst_element_link_many (streammux, queue1, primary_detector, queue2,
      nvtile, queue3, nvvidconv, queue4,
        nvosd, NULL)) {
    g_printerr ("Inferring and tracking elements link failure.\n");
    return -1;
  }

  g_object_set (G_OBJECT (sink), "sync", 0, "async", false,NULL);

  if (output_type == 1) {
    g_object_set (G_OBJECT (sink), "location", filepath, NULL);
    g_object_set (G_OBJECT (sink), "enable-last-sample", false,NULL);
    if (!isImage) {
      gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
          queue5, queue6, encparse, mux, NULL);
      if (!gst_element_link_many (nvosd, queue5, nvvidconv1, capfilt, queue6,
             outenc, encparse, mux, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
      }
    } else {
      gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
          queue5, queue6, NULL);
      if (!gst_element_link_many (nvosd, queue5, nvvidconv1, capfilt, queue6,
             outenc, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
      }
    }
    g_free(filepath);
  } else if (output_type==2) {
    if (!gst_element_link (nvosd, sink)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 3) {
    gst_bin_add (GST_BIN (pipeline), queue5);
    if (!gst_element_link_many (nvosd, queue5, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 4) {
    if (isImage) {
      g_printerr ("Can not output a image as RTSP stream.\n");
      return -1;
    }
    GstRTSPServer *server = NULL;
    GstRTSPMediaFactory *factory;

    server = gst_rtsp_server_new ();
    factory = gst_rtsp_media_factory_new ();

    if(isYAML) {
      isH264 = !(ds_parse_enc_codec(argv[1], "output"));
      enc_type = ds_parse_enc_type(argv[1], "output");
    }

    create_video_encoder(isH264, enc_type, &capfilt, &outenc, &encparse, &rtppay);
    gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
        queue5, queue6, encparse, rtppay, NULL);

    if (!gst_element_link_many (nvosd, queue5, nvvidconv1, capfilt, queue6,
           outenc, encparse, rtppay, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }

    if(isYAML) {
      ds_parse_enc_config (outenc, argv[1], "output");
      ds_parse_rtsp_output(sink, server, factory, argv[1], "output");
    } else {
      g_object_set (G_OBJECT (outenc), "bitrate", 4000000, NULL);
      g_object_set (G_OBJECT (sink), "host", "224.224.255.255", "port",
      atoi(argv[3]), "async", FALSE, "sync", 0, NULL);

      char udpsrc_pipeline[512];
      sprintf (udpsrc_pipeline,
        "( udpsrc name=pay0 port=%s buffer-size=65536 caps=\"application/x-rtp, media=video, "
        "clock-rate=90000, encoding-name=H264, payload=96 \" )",
        argv[3]);
      g_object_set (server, "service", argv[4], NULL);
      gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);
      g_print("Please reach RTSP with rtsp://ip:%s/ds-out-avc\n", argv[4]);
    }
    GstRTSPMountPoints *mounts;
    mounts = gst_rtsp_server_get_mount_points (server);
    gst_rtsp_mount_points_add_factory (mounts, "/ds-out-avc", factory);
    g_object_unref (mounts);

    gst_rtsp_server_attach (server, NULL);
  }

  /* Read cvcore parameters from config file.*/
  std::vector<std::pair<uint32_t, uint32_t>> connect_table;
  if(isYAML) {
    GString * temp = ds_parse_config_yml_filepath(argv[1], "model-config");

    YAML::Node config = YAML::LoadFile(temp->str);

    if (config.IsNull()) {
      g_printerr ("config file is NULL.\n");
      return -1;
    }

    if (config["nmsWindowSize"]) {
      postprocess_params_list["nmsWindowSize"] =
        config["nmsWindowSize"].as<float>();
    }
    if (config["featUpsamplingFactor"]) {
      postprocess_params_list["featUpsamplingFactor"] =
        config["featUpsamplingFactor"].as<float>();
    }
    if (config["threshHeat"]) {
      postprocess_params_list["threshHeat"] =
        config["threshHeat"].as<float>();
    }
    if (config["threshVectorScore"]) {
      postprocess_params_list["threshVectorScore"] =
        config["threshVectorScore"].as<float>();
    }
    if (config["threshVectorCnt1"]) {
      postprocess_params_list["threshVectorCnt1"] =
        config["threshVectorCnt1"].as<float>();
    }
    if (config["threshPartCnt"]) {
      postprocess_params_list["threshPartCnt"] =
        config["threshPartCnt"].as<float>();
    }
    if (config["threshHumanScore"]) {
      postprocess_params_list["threshHumanScore"] =
        config["threshHumanScore"].as<float>();
    }
    if (config["batchSize"]) {
      postprocess_params_list["batchSize"] =
        config["batchSize"].as<float>();
    }
    if (config["imageType"]) {
      postprocess_params_list["imageType"] =
        config["imageType"].as<float>();
    }
    if (config["numJoints"]) {
      postprocess_params_list["numJoints"] =
        config["numJoints"].as<float>();
    }
    if (config["jointEdges"]) {
      YAML::Node primes = config["jointEdges"];
      for (YAML::const_iterator it=primes.begin();it!=primes.end();++it) {

        std::string seq = it->as<std::string>();
        std::vector<std::string> token_vec;
        std::string token;
        std::stringstream ss(seq);
        while(std::getline(ss, token, ',')){
          gchar *str = (char *) token.c_str();
          token_vec.push_back(str);
        }
        int  vec_size = token_vec.size();
        if (vec_size != 2)
          continue;

        std::pair<uint32_t, uint32_t> edge_pair;
        std::istringstream x_pos_str(token_vec[0]);
        std::istringstream y_pos_str(token_vec[1]);
        x_pos_str >> edge_pair.first;
        y_pos_str >> edge_pair.second;
        connect_table.push_back(edge_pair);
        token_vec.clear();
      }
    }
    if (config["connections"]) {
      YAML::Node primes = config["connections"];
      for (YAML::const_iterator it=primes.begin();it!=primes.end();++it) {
        std::string seq = it->as<std::string>();
        std::vector<std::string> token_vec;
        std::string token;
        std::stringstream ss(seq);

        while(std::getline(ss, token, ',')){
          gchar *str = (char *) token.c_str();
          token_vec.push_back(str);
        }
        int  vec_size = token_vec.size();
        if (vec_size != 2)
          continue;

        std::istringstream x_pos_str(token_vec[0]);
        std::istringstream y_pos_str(token_vec[1]);
        std::pair<uint32_t, uint32_t> connection_pair;
        x_pos_str >> connection_pair.first;
        y_pos_str >> connection_pair.second;
        display_connect_table.push_back(connection_pair);
        token_vec.clear();
      }
    }
    g_string_free(temp, TRUE);
  } else {
    fconfig.open(argv[2]);
    if (!fconfig.is_open()) {
      g_print("The model config file open is failed!\n");
      return -1;
    }

    while (!fconfig.eof()) {
      string strParam;
	  if ( getline(fconfig, strParam) ) {
          std::vector<std::string> param_strs = split(strParam, "=");
          float value;
          if (param_strs.size() < 2)
              continue;
          if(!(param_strs[0].empty()) && !(param_strs[1].empty())) {
              if (param_strs[0] == "jointEdges") {
                  std::vector<std::string> edge_parse_strs = split(param_strs[1], ",");
                  if ( !(edge_parse_strs[0].empty())  && ! (edge_parse_strs[1].empty()) ) {
                      std::istringstream x_pos_str(edge_parse_strs[0]);
                      std::istringstream y_pos_str(edge_parse_strs[1]);
                      std::pair<uint32_t, uint32_t> edge_pair;
                      x_pos_str >> edge_pair.first;
                      y_pos_str >> edge_pair.second;

                      g_print("joint Edges %d , %d\n", edge_pair.first,edge_pair.second);
                      connect_table.push_back(edge_pair);
                  }
              } else if (param_strs[0] == "connections") {
                  std::vector<std::string> connection_parse_strs = split(param_strs[1], ",");
                  if ( !(connection_parse_strs[0].empty())  && ! (connection_parse_strs[1].empty()) ) {
                      std::istringstream x_pos_str(connection_parse_strs[0]);
                      std::istringstream y_pos_str(connection_parse_strs[1]);
                      std::pair<uint32_t, uint32_t> connection_pair;
                      x_pos_str >> connection_pair.first;
                      y_pos_str >> connection_pair.second;

                      g_print("connections %d , %d\n", connection_pair.first,connection_pair.second);
                      display_connect_table.push_back(connection_pair);
                  }
              }else {
                  std::istringstream isStr(param_strs[1]);
                  isStr >> value;
                  postprocess_params_list[param_strs[0]] = value;
              }
          }
	  }
    }
    fconfig.close();
  }

  cvcore::bodypose2d::BodyPose2DPostProcessorParams postProcessParams = {};
  cvcore::ModelInputParams ModelInputParams = {8, 384, 288, cvcore::RGB_U8};

  if (postprocess_params_list.count("threshVectorCnt1"))
      postProcessParams.threshVectorCnt1 = postprocess_params_list["threshVectorCnt1"];
  if (postprocess_params_list.count("nmsWindowSize"))
      postProcessParams.nmsWindowSize = postprocess_params_list["nmsWindowSize"];
  if (postprocess_params_list.count("featUpsamplingFactor"))
      postProcessParams.featUpsamplingFactor = postprocess_params_list["featUpsamplingFactor"];
  if (postprocess_params_list.count("threshHeat"))
      postProcessParams.threshHeat = postprocess_params_list["threshHeat"];
  if (postprocess_params_list.count("threshVectorScore"))
      postProcessParams.threshVectorScore = postprocess_params_list["threshVectorScore"];
  if (postprocess_params_list.count("threshPartCnt"))
      postProcessParams.threshPartCnt = postprocess_params_list["threshPartCnt"];
  if (postprocess_params_list.count("threshHumanScore"))
      postProcessParams.threshHumanScore = postprocess_params_list["threshHumanScore"];
  if (postprocess_params_list.count("batchSize"))
      ModelInputParams.maxBatchSize = postprocess_params_list["batchSize"];
  if (postprocess_params_list.count("imageType"))
      ModelInputParams.modelInputType = (cvcore::ImageType)postprocess_params_list["imageType"];
  if (postprocess_params_list.count("numJoints"))
      postProcessParams.numJoints = (size_t)postprocess_params_list["numJoints"];

  for (int cnt=0; cnt < connect_table.size(); cnt++) {
      if (connect_table[cnt].first <= postProcessParams.numJoints && connect_table[cnt].first >= 0
           && connect_table[cnt].second <= postProcessParams.numJoints && connect_table[cnt].second >= 0) {
          postProcessParams.jointEdges.push_back(connect_table[cnt]);
      } else {
          connect_table.erase(connect_table.begin() + cnt);
      }
  }
  for (int cnt=0; cnt < display_connect_table.size(); cnt++) {
      if (display_connect_table[cnt].first > postProcessParams.numJoints || display_connect_table[cnt].first < 0
           || display_connect_table[cnt].second > postProcessParams.numJoints || display_connect_table[cnt].second < 0) {
          display_connect_table.erase(display_connect_table.begin() + cnt);
      }
  }

  std::unique_ptr< cvcore::bodypose2d::BodyPose2DPostProcessor > body2Dposepostinit(
    new cvcore::bodypose2d::BodyPose2DPostProcessor (
    postProcessParams, ModelInputParams));
  body2Dposepost = std::move(body2Dposepostinit);

  /* Generate display meta for the bodypose2d output on video. */
  /*  Fakesink do not need to generate display meta.           */
  if (output_type != 2) {
    osd_sink_pad = gst_element_get_static_pad (nvtile, "sink");
    if (!osd_sink_pad)
      g_print ("Unable to get sink pad\n");
    else
      gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
          tile_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);
  }

  /*Performance measurement*/
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, &perf_measure, NULL);
  gst_object_unref (osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  osd_sink_pad = gst_element_get_static_pad (queue2, "src");
  if (!osd_sink_pad)
    g_print ("Unable to get nvinfer src pad\n");
  gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);  

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[5]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);

  if(perf_measure.total_time)
  {
    g_print ("Average fps %f\n",
      ((perf_measure.count-1)*src_cnt*1000000.0)/perf_measure.total_time);
  }

  g_print ("Totally %d persons are inferred\n",total_person_num);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
