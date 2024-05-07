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
#include "ds_facialmark_meta.h"
#include "cv/core/Tensor.h"
#include "nvbufsurface.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"
#include <yaml-cpp/yaml.h>

using namespace std;
using std::string;

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_FACE 0

#define PGIE_DETECTED_CLASS_NUM 4

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define CONFIG_GPU_ID "gpu-id"

#define SGIE_NET_WIDTH 80
#define SGIE_NET_HEIGHT 80

gint frame_number = 0;
gint total_face_num = 0;

#define PRIMARY_DETECTOR_UID 1
#define SECOND_DETECTOR_UID 2

std::unique_ptr<cvcore::faciallandmarks::FacialLandmarksPostProcessor> facemarkpost;

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

/* Calculate performance data */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsObjectMeta *obj_meta = NULL;
  guint face_count = 0;
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
        
      /* Check that the object has been detected by the primary detector
      * and that the class id is that of vehicles/persons. */
      if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
        if (obj_meta->class_id == PGIE_CLASS_ID_FACE)
          face_count++;
      }
    }
  }

  g_print ("Frame Number = %d Face Count = %d\n",
           frame_number, face_count);
  frame_number++;
  total_face_num += face_count;
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

      bool facebboxdraw = false;
        
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
          l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if(user_meta->base_meta.meta_type ==
            (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {
          NvDsFacePointsMetaData *facepoints_meta =
              (NvDsFacePointsMetaData *)user_meta->user_meta_data;
          /*Get the face marks and mark with dots*/
          if (!facepoints_meta)
            continue;
          for (part_index = 0;part_index < facepoints_meta->facemark_num;
              part_index++) {
            if (!disp_meta) {
              disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
              disp_meta->num_circles = 0;
              disp_meta->num_rects = 0;
              
            } else {
              if (disp_meta->num_circles==MAX_ELEMENTS_IN_DISPLAY_META) {
                
                nvds_add_display_meta_to_frame (frame_meta, disp_meta);
                disp_meta = nvds_acquire_display_meta_from_pool(batch_meta);
                disp_meta->num_circles = 0;
              }
            }
            if(!facebboxdraw) {
                disp_meta->rect_params[disp_meta->num_rects].left =
                    facepoints_meta->right_eye_rect.left +
                    obj_meta->rect_params.left;
                disp_meta->rect_params[disp_meta->num_rects].top =
                    facepoints_meta->right_eye_rect.top +
                    obj_meta->rect_params.top;
                disp_meta->rect_params[disp_meta->num_rects].width =
                    facepoints_meta->right_eye_rect.right -
                    facepoints_meta->right_eye_rect.left;
                disp_meta->rect_params[disp_meta->num_rects].height =
                    facepoints_meta->right_eye_rect.bottom -
                    facepoints_meta->right_eye_rect.top;
                disp_meta->rect_params[disp_meta->num_rects].border_width = 2;
                disp_meta->rect_params[disp_meta->num_rects].border_color.red = 1.0;
                disp_meta->rect_params[disp_meta->num_rects].border_color.green = 1.0;
                disp_meta->rect_params[disp_meta->num_rects].border_color.blue = 0.0;
                disp_meta->rect_params[disp_meta->num_rects].border_color.alpha = 0.5;
                disp_meta->rect_params[disp_meta->num_rects+1].left =
                    facepoints_meta->left_eye_rect.left + obj_meta->rect_params.left;
                disp_meta->rect_params[disp_meta->num_rects+1].top =
                    facepoints_meta->left_eye_rect.top + obj_meta->rect_params.top;
                disp_meta->rect_params[disp_meta->num_rects+1].width =
                    facepoints_meta->left_eye_rect.right -
                    facepoints_meta->left_eye_rect.left;
                disp_meta->rect_params[disp_meta->num_rects+1].height =
                    facepoints_meta->left_eye_rect.bottom -
                    facepoints_meta->left_eye_rect.top;
                disp_meta->rect_params[disp_meta->num_rects+1].border_width = 2;
                disp_meta->rect_params[disp_meta->num_rects+1].border_color.red = 1.0;
                disp_meta->rect_params[disp_meta->num_rects+1].border_color.green = 1.0;
                disp_meta->rect_params[disp_meta->num_rects+1].border_color.blue = 0.0;
                disp_meta->rect_params[disp_meta->num_rects+1].border_color.alpha = 0.5;
                disp_meta->num_rects+=2;
                facebboxdraw = true;
            }

            disp_meta->circle_params[disp_meta->num_circles].xc =
                facepoints_meta->mark[part_index].x + obj_meta->rect_params.left;
            disp_meta->circle_params[disp_meta->num_circles].yc =
                facepoints_meta->mark[part_index].y + obj_meta->rect_params.top;
            disp_meta->circle_params[disp_meta->num_circles].radius = 1;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.red = 0.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.green = 1.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.blue = 0.0;
            disp_meta->circle_params[disp_meta->num_circles].circle_color.alpha = 0.5;
            disp_meta->num_circles++;
          }
        }
      }
    }
    if (disp_meta && disp_meta->num_circles)
       nvds_add_display_meta_to_frame (frame_meta, disp_meta);
  }
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. The face bbox will be scale to square for
 * facial marks.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));
  NvBufSurface *in_surf;
  GstMapInfo in_map_info;
  int frame_width, frame_height;

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
    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {

      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      
      if (!obj_meta) {
        g_print("No obj meta\n");
        break;
      }
      if(obj_meta->rect_params.left<0)
          obj_meta->rect_params.left=0;
      if(obj_meta->rect_params.top<0)
          obj_meta->rect_params.top=0;
          
      float square_size = MAX(obj_meta->rect_params.width,
          obj_meta->rect_params.height);
      float center_x = obj_meta->rect_params.width/2.0 +
          obj_meta->rect_params.left;
      float center_y = obj_meta->rect_params.height/2.0 +
          obj_meta->rect_params.top;

      /*Check the border*/
      if(center_x < (square_size/2.0) || center_y < square_size/2.0 || 
          center_x + square_size/2.0 > frame_width ||
          center_y - square_size/2.0 > frame_height) {
              g_print("Keep the original bbox\n");
      } else {
          obj_meta->rect_params.left = center_x - square_size/2.0;
          obj_meta->rect_params.top = center_y - square_size/2.0;
          obj_meta->rect_params.width = square_size;
          obj_meta->rect_params.height = square_size;
      }
    }
  }
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the src pad
 * of the SGIE's next queue element. The facial marks output will be processed
 * and the facial marks metadata will be generated.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    //NvDsDisplayMeta *disp_meta = NULL;
    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {

      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      if (!obj_meta)
        continue;

      /* Iterate user metadata in object to search SGIE's tensor data */
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;
        float * heatmap_data = NULL;
        float * confidence = NULL;
        //int heatmap_c = 0;

        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          info->buffer = meta->out_buf_ptrs_host[i];

          std::vector < NvDsInferLayerInfo >
            outputLayersInfo (meta->output_layers_info,
            meta->output_layers_info + meta->num_output_layers);
          //Prepare CVCORE input layers
          if (strcmp(outputLayersInfo[i].layerName,
              "softargmax") == 0) {
            //This layer output landmarks coordinates
            heatmap_data = (float *)meta->out_buf_ptrs_host[i];
          } else if (strcmp(outputLayersInfo[i].layerName,
              "softargmax:1") == 0) {
            confidence = (float *)meta->out_buf_ptrs_host[i];
          }
        }

        cvcore::Tensor<cvcore::CL, cvcore::CX, cvcore::F32> tempheatmap(
            cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS, 1,
            (float *)heatmap_data, true);
        cvcore::Array<cvcore::BBox> faceBBox(1);
        faceBBox.setSize(1);
        faceBBox[0] = {0, 0, (int)obj_meta->rect_params.width,
            (int)obj_meta->rect_params.height};
        //Prepare output array
        cvcore::Array<cvcore::ArrayN<cvcore::Vector2f,
            cvcore::faciallandmarks::FacialLandmarks::MAX_NUM_FACIAL_LANDMARKS>>
            output(1, true);
        output.setSize(1);
      
        facemarkpost->execute(output, tempheatmap, faceBBox, NULL);
      
        /*add user meta for facemark*/
        if (!nvds_add_facemark_meta (batch_meta, obj_meta, output[0],
            confidence)) {
          g_printerr ("Failed to get bbox from model output\n");
        }
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
        g_printerr ("Failed to link decoderbin src pad to"
            " converter sink pad\n");
      }
      g_object_unref(conv_sink_pad);
      if (!gst_element_link (bin_struct->vidconv, bin_struct->nvvidconv)) {
         g_printerr ("Failed to link videoconvert to nvvideoconvert\n");
      }
    } else {
      GstPad *conv_sink_pad = gst_element_get_static_pad (bin_struct->nvvidconv,
          "sink");
      if (gst_pad_link (decoder_src_pad, conv_sink_pad)) {
        g_printerr ("Failed to link decoderbin src pad to "
            "converter sink pad\n");
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
  } else {
    bin_struct->vidconv = NULL;
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

  if (!ds_source_struct->source_bin || !ds_source_struct->uri_decode_bin ||
      !ds_source_struct->nvvidconv
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
      gst_ghost_pad_new ("src", gstpad))) {
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
             *primary_detector = NULL, *second_detector = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL, *nvtile=NULL,
             *mux = NULL, *encparse = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL, *queue7 = NULL;
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
  ifstream fconfig;
  std::map<string, float> postprocess_params_list;
  bool isYAML=false;
  bool isImage=false;
  bool isStreaming=false;
  GList* g_list = NULL;
  GList* iterator = NULL;
  bool isH264 = true;
  int enc_type = ENCODER_TYPE_HW;
  gchar *filepath = NULL;

  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER, sgie_type = NVDS_GIE_PLUGIN_INFER;
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
    if(nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie") == NVDS_YAML_PARSER_SUCCESS) {
      g_print("sgie_type %d\n", sgie_type);
    }
  } else {
    if (argc < 5 || argc > 132 || (atoi(argv[1]) != 1 && atoi(argv[1]) != 2 &&
      atoi(argv[1]) != 3)) {
      g_printerr ("Usage: %s [1:file sink|2:fakesink|3:display sink] "
        "<faciallandmark model config file> "
        "<input uri> ... <input uri> <out H264 filename>\n "
	"OR\n %s <app YAML config file>\n", argv[0], argv[0]);
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
    for (src_cnt=0; src_cnt<(guint)argc-4; src_cnt++) {
      g_list = g_list_append(g_list, argv[src_cnt + 3]);
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
    if (g_strrstr ((gchar *)iterator->data, ".jpg") ||
        g_strrstr ((gchar *)iterator->data, ".jpeg") ||
        g_strrstr ((gchar *)iterator->data, ".png"))
      isImage = true;
    else
      isImage = false;
    if (g_strrstr ((gchar *)iterator->data, "rtsp://") ||
        g_strrstr ((gchar *)iterator->data, "v4l2://") ||
        g_strrstr ((gchar *)iterator->data, "http://") ||
        g_strrstr ((gchar *)iterator->data, "rtmp://")) {
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

  /* Create three nvinfer instances for two detectors. */
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    primary_detector = gst_element_factory_make ("nvinferserver",
                         "primary-infer-engine1");
  } else {
    primary_detector = gst_element_factory_make ("nvinfer",
                         "primary-infer-engine1");
  }
  if (sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
    second_detector = gst_element_factory_make ("nvinferserver",
                         "second-infer-engine");
  } else {
    second_detector = gst_element_factory_make ("nvinfer",
                         "second-infer-engine");
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
  queue7 = gst_element_factory_make ("queue", "queue7");

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
    }
    else {
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

  if (!primary_detector || !second_detector || !nvvidconv
      || !nvosd || !sink  || !capfilt) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if (isYAML)
      nvds_parse_streammux(streammux, argv[1], "streammux");
  else
      g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  if (isStreaming)
    g_object_set (G_OBJECT (streammux), "live-source", true, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", src_cnt, NULL);

  tiler_rows = (guint) sqrt (src_cnt);
  tiler_columns = (guint) ceil (1.0 * src_cnt / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", 1280, "height", 720, NULL);

  /* Set the config files for the two detectors. The first detector is PGIE which
   * detects the faces. The second detector is SGIE which generates faciallandmarks
   * for every face. */
  if(isYAML)
    nvds_parse_gie (primary_detector, argv[1], "primary-gie");
  else
    g_object_set (G_OBJECT (primary_detector), "config-file-path",
      "../../../configs/nvinfer/facial_tao/config_infer_primary_facenet.txt",
      "unique-id", PRIMARY_DETECTOR_UID, NULL);

  if(isYAML)
    nvds_parse_gie (second_detector, argv[1], "secondary-gie");
  else
    g_object_set (G_OBJECT (second_detector), "config-file-path",
      "../../../configs/nvinfer/facial_tao/faciallandmark_sgie_config.txt",
      "unique-id", SECOND_DETECTOR_UID, NULL);

  /* we add a bus message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), primary_detector, second_detector,
      queue1, queue2, queue3, queue4, queue5, nvvidconv, nvosd, nvtile, sink,
      NULL);

  if (!gst_element_link_many (streammux, queue1, primary_detector, queue2, 
        second_detector, queue3, nvtile, queue4, nvvidconv, queue5,
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
        queue6, queue7, encparse, mux, NULL);
      if (!gst_element_link_many (nvosd, queue6, nvvidconv1, capfilt, queue7,
           outenc, encparse, mux, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
      }
    } else {
      gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
        queue6, queue7, NULL);
      if (!gst_element_link_many (nvosd, queue6, nvvidconv1, capfilt, queue7,
            outenc, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
      }
    }
    g_free(filepath);
  } else if (output_type == 2) {
    if (!gst_element_link (nvosd, sink)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 3) {
    gst_bin_add (GST_BIN (pipeline), queue6);
    if (!gst_element_link_many (nvosd, queue6, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
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

    if (config["numLandmarks"]) {
      postprocess_params_list["numLandmarks"] =
        config["numLandmarks"].as<float>();
    }
    if (config["maxBatchSize"]) {
      postprocess_params_list["maxBatchSize"] =
        config["maxBatchSize"].as<float>();
    }
    if (config["inputLayerWidth"]) {
      postprocess_params_list["inputLayerWidth"] =
        config["inputLayerWidth"].as<float>();
    }
    if (config["inputLayerHeight"]) {
      postprocess_params_list["inputLayerHeight"] =
        config["inputLayerHeight"].as<float>();
    }
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
              std::istringstream isStr(param_strs[1]);
              isStr >> value;
              postprocess_params_list[param_strs[0]] = value;
          }
	  }
    }
    fconfig.close();
  }

  size_t numFaciallandmarks = 80;
  cvcore::ModelInputParams ModelInputParams = {32, 80, 80, cvcore::Y_F32};

  if (postprocess_params_list.count("maxBatchSize"))
      ModelInputParams.maxBatchSize = (size_t)postprocess_params_list["maxBatchSize"];
  if (postprocess_params_list.count("inputLayerWidth"))
      ModelInputParams.inputLayerWidth = (size_t)postprocess_params_list["inputLayerWidth"];
  if (postprocess_params_list.count("inputLayerHeight"))
      ModelInputParams.inputLayerHeight = (size_t)postprocess_params_list["inputLayerHeight"];
  if (postprocess_params_list.count("numLandmarks"))
      numFaciallandmarks = (size_t)postprocess_params_list["numLandmarks"];

 std::unique_ptr< cvcore::faciallandmarks::FacialLandmarksPostProcessor > faciallandmarkpostinit(
    new cvcore::faciallandmarks::FacialLandmarksPostProcessor (
    ModelInputParams,numFaciallandmarks));
  facemarkpost = std::move(faciallandmarkpostinit);

  /* Display the facemarks output on video. Fakesink do not need to display. */
  if(output_type != 2) {    
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

  /* Add probe to get square bbox from face detector. */
  osd_sink_pad = gst_element_get_static_pad (queue2, "src");
  if (!osd_sink_pad)
    g_print ("Unable to get nvinfer src pad\n");
  gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Add probe to handle facial marks output and generate facial */
  /* marks metadata.                                             */
  osd_sink_pad = gst_element_get_static_pad (queue3, "src");
  if (!osd_sink_pad)
    g_print ("Unable to get nvinfer src pad\n");
  gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);
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

  g_print ("Totally %d faces are inferred\n",total_face_num);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
