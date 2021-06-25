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

#define BODY_CONNECT_NUM 17

gint frame_number = 0;
gint total_person_num = 0;

/* 0 - nose
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
const int connect_table[BODY_CONNECT_NUM][2] = {{0,1}, {1,2}, {1,5}, {2,3},
                                                {3,4}, {5,6}, {6,7},{2,8},
                                                {8,9}, {9,10}, {5,11},{11,12},
                                                {12,13},{0,14},{14,16},{8,11},
                                                {15,17}};

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
    gint index;
    bool is_imagedec;
}DsSourceBinStruct;

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
            (NvDsMetaType) NVDS_USER_JARVIS_META_2DPOSE) {
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
          for (line_num = 0; line_num < BODY_CONNECT_NUM; line_num++) {
            if ((bodypart_map.count(connect_table[line_num][0]))
                && (bodypart_map.count(connect_table[line_num][1]))) {
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
                  bodypart_map[connect_table[line_num][0]]->x;
              disp_meta->line_params[disp_meta->num_lines].y1 =
                  bodypart_map[connect_table[line_num][0]]->y;
              disp_meta->line_params[disp_meta->num_lines].x2 =
                  bodypart_map[connect_table[line_num][1]]->x;
              disp_meta->line_params[disp_meta->num_lines].y2 =
                  bodypart_map[connect_table[line_num][1]]->y;
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
      } else {
          post_w = (frame_height * meta->network_info.width +
              meta->network_info.height/2)/meta->network_info.height;
          post_h = frame_height;
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
        rect_params.left = human.boundingBox.xmin;
        rect_params.top = human.boundingBox.ymin;
        rect_params.width = human.boundingBox.xmax - human.boundingBox.xmin;
        rect_params.height = human.boundingBox.ymax - human.boundingBox.ymin;
        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

        /*add user meta for 2dpose*/
        if (!nvds_add_2dpose_meta (batch_meta, obj_meta, human, frame_width,
          frame_height)) {
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
      if (!gst_element_link (bin_struct->vidconv, bin_struct->nvvidconv)) {
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

#ifndef PLATFORM_TEGRA
  g_object_set (G_OBJECT (ds_source_struct->nvvidconv),
      "nvbuf-memory-type", 3, NULL);
#endif

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
  GstElement *pipeline = NULL,*streammux = NULL, *sink = NULL, 
             *primary_detector = NULL, *nvtile = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL;
  DsSourceBinStruct source_struct[128];
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  //int i;
  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;
  perf_measure perf_measure;

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";
  gchar *filename;
  ifstream fconfig;
  std::map<string, float> postprocess_params_list;
    
  /* Check input arguments */
  if (argc < 5 || argc > 132 || (atoi(argv[1]) != 1 && atoi(argv[1]) != 2 &&
      atoi(argv[1]) != 3)) {
    g_printerr ("Usage: %s [1:file sink|2:fakesink|3:display sink] "
      "<model configure file> "
      "<input file> ... <inputfile> <out H264 filename>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
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
  for (src_cnt=0; src_cnt<(guint)argc-4; src_cnt++) {
    /* Source element for reading from the file */
    source_struct[src_cnt].index = src_cnt;
    if (!create_source_bin (&(source_struct[src_cnt]), argv[src_cnt + 3]))
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
  primary_detector = gst_element_factory_make ("nvinfer",
                       "primary-infer-engine1");

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

  if (atoi(argv[1]) == 1) {
    if (g_strrstr (argv[3], ".jpg") || g_strrstr (argv[3], ".png")
       || g_strrstr (argv[3], ".jpeg")) {
        outenc = gst_element_factory_make ("jpegenc", "jpegenc");
        caps =
            gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
            "I420", NULL);
        g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
        filename = g_strconcat(argv[argc-1],".jpg",NULL);
    }
    else {
        outenc = gst_element_factory_make ("nvv4l2h264enc" ,"nvvideo-h264enc");
        caps =
            gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
            "I420", NULL);
        feature = gst_caps_features_new ("memory:NVMM", NULL);
        gst_caps_set_features (caps, 0, feature);
        g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
        g_object_set (G_OBJECT (outenc), "bitrate", 4000000, NULL);
        filename = g_strconcat(argv[argc-1],".264",NULL);
    }
    sink = gst_element_factory_make ("filesink", "nvvideo-renderer");
  }
  else if (atoi(argv[1]) == 2)
    sink = gst_element_factory_make ("fakesink", "fake-renderer");
  else if (atoi(argv[1]) == 3) {
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegltransform");
    if(!transform) {
      g_printerr ("nvegltransform element could not be created. Exiting.\n");
      return -1;
    }
#endif
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  }

  if (!primary_detector || !nvvidconv
      || !nvosd || !sink  || !capfilt ) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", src_cnt,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  tiler_rows = (guint) sqrt (src_cnt);
  tiler_columns = (guint) ceil (1.0 * src_cnt / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", 1280, "height", 720, NULL);

  g_object_set (G_OBJECT (primary_detector), "config-file-path",
      "../../../configs/bodypose2d_tlt/bodypose2d_pgie_config.txt",
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

  if (atoi(argv[1]) == 1) {
    g_object_set (G_OBJECT (sink), "location", filename,NULL);
    g_object_set (G_OBJECT (sink), "enable-last-sample", false,NULL);
    gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
        queue5, queue6, NULL);

    if (!gst_element_link_many (nvosd, queue5, nvvidconv1, capfilt, queue6,
           outenc, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
    g_free(filename);
  } else if (atoi(argv[1]) == 2) {
    if (!gst_element_link (nvosd, sink)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (atoi(argv[1]) == 3) {
#ifdef PLATFORM_TEGRA
    gst_bin_add_many (GST_BIN (pipeline), transform, queue5, NULL);
    if (!gst_element_link_many (nvosd, queue5, transform, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
#else
    gst_bin_add (GST_BIN (pipeline), queue5);
    if (!gst_element_link_many (nvosd, queue5, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
#endif
  }

  /* Read cvcore parameters from config file.*/
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
              std::istringstream isStr(param_strs[1]);
              isStr >> value;
              postprocess_params_list[param_strs[0]] = value;
          }
	  }
  }
  fconfig.close();

  cvcore::bodypose2d::BodyPose2DPostProcessorParams postProcessParams =
                                          {38, 192,  144,  19, 48, 36, 7,
                                           4,  0.05, 0.05, 8,  4,  0.4};
  cvcore::ModelInputParams ModelInputParams = {8, 384, 288, cvcore::RGB_U8};

  if (postprocess_params_list.count("pafMapWidth"))
      postProcessParams.pafMapWidth = postprocess_params_list["pafMapWidth"];
  if (postprocess_params_list.count("pafMapHeight")) {
      postProcessParams.pafMapHeight = postprocess_params_list["pafMapHeight"];
      ModelInputParams.inputLayerWidth = postprocess_params_list["pafMapHeight"]*2;
  }
  if (postprocess_params_list.count("pafMapChannels")) {
      postProcessParams.pafMapChannels = postprocess_params_list["pafMapChannels"];
      ModelInputParams.inputLayerHeight = postprocess_params_list["pafMapChannels"] * 2;
  }
  if (postprocess_params_list.count("heatMapWidth"))
      postProcessParams.heatMapWidth = postprocess_params_list["heatMapWidth"];
  if (postprocess_params_list.count("heatMapHeight"))
      postProcessParams.heatMapHeight = postprocess_params_list["heatMapHeight"];
  if (postprocess_params_list.count("threshVectorCnt1"))
      postProcessParams.threshVectorCnt1 = postprocess_params_list["threshVectorCnt1"];
  if (postprocess_params_list.count("heatMapChannels"))
      postProcessParams.heatMapChannels = postprocess_params_list["heatMapChannels"];
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
      ModelInputParams.modelInputType = (cvcore::ImageType)postprocess_params_list["imageType"];;

  std::unique_ptr< cvcore::bodypose2d::BodyPose2DPostProcessor > body2Dposepostinit(
    new cvcore::bodypose2d::BodyPose2DPostProcessor (
    postProcessParams,
    ModelInputParams));
  body2Dposepost = std::move(body2Dposepostinit);

  /* Generate display meta for the bodypose2d output on video. */
  /*  Fakesink do not need to generate display meta.           */
  if(atoi(argv[1]) != 2) {    
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
  g_print ("Now playing: %s\n", argv[3]);
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
