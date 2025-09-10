/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <bits/stdc++.h>

#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "deepstream_common.h"
#include "deepstream_perf.h"
#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"

#define MAX_DISPLAY_LEN 64

// Default camera attributes
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Padding due to AR SDK model requires bigger bboxes*/
#define PAD_DIM 128

#define PGIE_CLASS_ID_PERSON 0

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr) \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) { \
    g_printerr("Error in parsing configuration file.\n"); \
    return -1; \
  }

//---Global variables derived from program arguments---
static guint _cintr = FALSE;
static gboolean _quit = FALSE;
int _image_width = MUXER_OUTPUT_WIDTH;
int _image_height = MUXER_OUTPUT_HEIGHT;
int _pad_dim = PAD_DIM;// A scaled version of PAD_DIM

static GstElement *pipeline = NULL;

gint frame_number = 0;

#define ACQUIRE_DISP_META(dmeta)  \
  if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META  || \
      dmeta->num_labels == MAX_ELEMENTS_IN_DISPLAY_META ||  \
      dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META) \
        { \
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);\
          nvds_add_display_meta_to_frame(frame_meta, dmeta);\
        }\

static float _sgie_classifier_threshold = FLT_MIN;


/* pgie_src_pad_buffer_probe will extract metadata received from pgie
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
pgie_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  gchar *msg = NULL;
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_user = NULL;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next)
    {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
      float sizex = obj_meta->rect_params.width * .5f;
      float sizey = obj_meta->rect_params.height * .5f;
      float centrx = obj_meta->rect_params.left  + sizex;
      float centry = obj_meta->rect_params.top  + sizey;
      sizex *= (1.25f);
      sizey *= (1.25f);
      if (sizex < sizey)
        sizex = sizey;
      else
        sizey = sizex;

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);

      sizex= obj_meta->rect_params.width * .5f, sizey = obj_meta->rect_params.height * .5f;
      centrx = obj_meta->rect_params.left + sizex, centry = obj_meta->rect_params.top + sizey;
      // Make sure box has same aspect ratio as 3D Body Pose model's input dimensions
      // (e.g 192x256 -> 0.75 aspect ratio) by enlarging in the appropriate dimension.
      float xScale = (float)192.0 / (float)sizex, yScale = (float)256.0 / (float)sizey;
      if (xScale < yScale) { // expand on height
          sizey = (float)256.0/ xScale;
      }
      else { // expand on width
          sizex = (float)192.0 / yScale;
      }

      obj_meta->rect_params.width = roundf(2.f *sizex);
      obj_meta->rect_params.height = roundf(2.f *sizey);
      obj_meta->rect_params.left   = roundf (centrx - obj_meta->rect_params.width/2.f);
      obj_meta->rect_params.top    = roundf (centry - obj_meta->rect_params.height/2.f);
      if (obj_meta->rect_params.left < 0.0) {
          obj_meta->rect_params.left = 0.0;
      }
      if (obj_meta->rect_params.top < 0.0) {
        obj_meta->rect_params.top = 0.0;
      }
      if (obj_meta->rect_params.left + obj_meta->rect_params.width > muxer_output_width_pad -1){
        obj_meta->rect_params.width = muxer_output_width_pad - 1 - obj_meta->rect_params.left;
      }
      if (obj_meta->rect_params.top + obj_meta->rect_params.height > muxer_output_height_pad -1){
        obj_meta->rect_params.height = muxer_output_height_pad - 1 - obj_meta->rect_params.top;
      }

    }
  }
  return GST_PAD_PROBE_OK;
}

/* osd_sink_pad_buffer_probe  will extract metadata received from OSD
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                          gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *)info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next)
  {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
    {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);
    }
    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);

    /* Parameters to draw text onto the On-Screen-Display */
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Frame Number %d", frame_number);
    offset = snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN, " ");

    txt_params->x_offset = 50;
    txt_params->y_offset = 100;

    char font_name[] = "Mono";
    txt_params->font_params.font_name = font_name;
    txt_params->font_params.font_size = 15;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }
  frame_number++;
  return GST_PAD_PROBE_OK;
}

typedef struct _DsSourceBin
{
    GstElement *source_bin;
    GstElement *uri_decode_bin;
    GstElement *vidconv;
    GstElement *nvvidconv;
    GstElement *capsfilt;
    GstElement *capsraw;
    gint index;
}DsSourceBinStruct;

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
      if (!gst_element_link_many (bin_struct->vidconv, bin_struct->capsraw,
         bin_struct->nvvidconv, NULL)) {
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
    if (gst_caps_features_contains (features, "memory:NVMM")) {
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

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler (int signum)
{
  struct sigaction action;

  NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  _cintr = TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (_quit) {
    return FALSE;
  }

  if (_cintr) {
    _cintr = FALSE;

    _quit = TRUE;
    GMainLoop *loop = (GMainLoop *) data;
    g_main_loop_quit (loop);

    return FALSE;
  }
  return TRUE;
}

int main(int argc, char *argv[])
{
  guint num_sources = 0;

  GMainLoop *loop = NULL;
  GstCaps *caps = NULL;
  GstElement *streammux = NULL, *pgie = NULL, *sgie = NULL, *postprocess = NULL, *preprocess1 = NULL, *sgie1 = NULL;;
  // Padding the image and removing the padding
  GstElement *nvvideoconvert_enlarge = NULL, *nvvideoconvert_reduce = NULL,
    *capsFilter_enlarge = NULL, *capsFilter_reduce = NULL;
  GstElement *nvvidconv = NULL, *nvtile = NULL, *nvosd = NULL, *tracker = NULL, *nvdslogger = NULL;
  GstElement *sink = NULL;
  DsSourceBinStruct source_struct[128];
  GstBus *bus = NULL;
  guint bus_watch_id;

  gboolean useDisplay = FALSE;
  gboolean useFakeSink = FALSE;
  gboolean useFileSink = FALSE;
  guint tiler_rows, tiler_columns;
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isStreaming=false;
  GList* g_list = NULL;
  GList* iterator = NULL;
  bool isH264 = true;
  gchar *filepath = NULL;


  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Standard GStreamer initialization */
  // signal(SIGINT, sigintHandler);
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, NULL);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("deepstream_pose_classfication_app");
  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "streammux-pgie");
  if (!streammux) {
    g_printerr ("PGIE streammux could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add(GST_BIN(pipeline), streammux);

  parse_streammux_width_height_yaml(&_image_width, &_image_height, argv[1]);
  g_print("width %d hight %d\n", _image_width, _image_height);

  _pad_dim = PAD_DIM * _image_width / MUXER_OUTPUT_WIDTH;
  //---Set properties of streammux---

  if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
    g_printerr ("No source is found. Exiting.\n");
    return -1;
  }

  for (iterator = g_list, num_sources=0; iterator; iterator = iterator->next,num_sources++) {
    /* Source element for reading from the file */
    source_struct[num_sources].index = num_sources;

    if (g_strrstr ((gchar *)iterator->data, "rtsp://") ||
        g_strrstr ((gchar *)iterator->data, "v4l2://") ||
        g_strrstr ((gchar *)iterator->data, "http://") ||
        g_strrstr ((gchar *)iterator->data, "rtmp://")) {
      isStreaming = true;
    } else {
      isStreaming = false;
    }

    g_print("video %s\n", (gchar *)iterator->data);

    if (!create_source_bin (&(source_struct[num_sources]), (gchar *)iterator->data))
    {
      g_printerr ("Source bin could not be created. Exiting.\n");
      return -1;
    }
      
    gst_bin_add (GST_BIN (pipeline), source_struct[num_sources].source_bin);
      
    g_snprintf (pad_name_sink, 64, "sink_%d", num_sources);
    sinkpad = gst_element_request_pad_simple (streammux, pad_name_sink);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_struct[num_sources].source_bin,
        pad_name_src);
    if (!srcpad) {
      g_printerr ("Decoder request src pad failed. Exiting.\n");
      return -1;
    }
    GstPadLinkReturn ret = gst_pad_link (srcpad, sinkpad);
    if ( ret != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting. %d\n",ret);
      return -1;
    }
    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);
  }

  nvds_parse_streammux(streammux, argv[1], "streammux");

  if (isStreaming)
    g_object_set (G_OBJECT (streammux), "live-source", true, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      pgie = gst_element_factory_make("nvinferserver", "primary-nvinference-engine");
  } else {
      pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
  }
  if (!pgie) {
    g_printerr ("PGIE element could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_gie (pgie, argv[1], "primary-gie");

  /* Override the batch-size set in the config file with the number of sources. */
  guint pgie_batch_size = 0;
  g_object_get(G_OBJECT(pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);

    g_object_set(G_OBJECT(pgie), "batch-size", num_sources, NULL);
  }

  //---Set pgie properties---

  /* We need to have a tracker to track the identified objects */
  tracker = gst_element_factory_make ("nvtracker", "tracker");
  if (!tracker) {
    g_printerr ("Nvtracker could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_tracker(tracker, argv[1], "tracker");

  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
  if (!nvdslogger) {
      g_printerr ("Nvdslogger could not be created. Exiting.\n");
      return -1;
  }
  g_object_set (G_OBJECT(nvdslogger), "fps-measurement-interval-sec",
        1, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* pgie_src_pad = gst_element_get_static_pad(tracker, "src");
  if (!pgie_src_pad)
    g_printerr ("Unable to get src pad for pgie\n");
  else
    gst_pad_add_probe(pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pgie_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (pgie_src_pad);

  /* 3d bodypose secondary gie */
  NvDsGieType sgie0_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie0_type, argv[1], "secondary-gie0"));
  if (sgie0_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      sgie = gst_element_factory_make("nvinferserver", "secondary-nvinference-engine");
  } else {
      sgie = gst_element_factory_make("nvinfer", "secondary-nvinference-engine");
  }
  if (!sgie) {
    g_printerr ("Secondary nvinfer could not be created. Exiting.\n");
    return -1;
  }
  //---Set sgie properties---
  /* Configure the nvinfer element using the nvinfer config file. */
  nvds_parse_gie (sgie, argv[1], "secondary-gie0");

  /* Override the batch-size set in the config file with the number of sources. */
  guint sgie_batch_size = 0;
  g_object_get(G_OBJECT(sgie), "batch-size", &sgie_batch_size, NULL);
  if (sgie_batch_size < num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        sgie_batch_size, num_sources);

    g_object_set(G_OBJECT(sgie), "batch-size", num_sources, NULL);
  }
  //---Set sgie properties---

  /* postprocess for 3d bodypose secondary gie */
  postprocess = gst_element_factory_make("nvdspostprocess", "postprocess-plugin");
  nvds_parse_postprocess(postprocess, argv[1], "secondary-postprocess0");

  /* preprocess + bodypose classification */
  preprocess1 = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");
  nvds_parse_preprocess(preprocess1, argv[1], "secondary-preprocess1");
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;
  RETURN_ON_PARSER_ERROR(nvds_parse_gie_type(&sgie1_type, argv[1], "secondary-gie1"));
  if (sgie1_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
      sgie1 = gst_element_factory_make("nvinferserver", "bodypose-classification-nvinference-engine");
  } else {
      sgie1 = gst_element_factory_make("nvinfer", "bodypose-classification-nvinference-engine");
  }
  if (!sgie1) {
    g_printerr ("sgie1 could not be created. Exiting.\n");
    return -1;
  }
  nvds_parse_gie (sgie1, argv[1], "secondary-gie1");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
  if (!nvvidconv) {
    g_printerr ("nvvidconv could not be created. Exiting.\n");
    return -1;
  }

  //---Manipulate image size so that PGIE bbox is large enough---
  // Enlarge image so that PeopleNet detected bbox is larger which would fully cover the
  // detected object in the original sized image.
  nvvideoconvert_enlarge = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_enlarge");
  if (!nvvideoconvert_enlarge) {
    g_printerr ("nvvideoconvert_enlarge could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_enlarge = gst_element_factory_make("capsfilter", "capsFilter_enlarge");
  if (!capsFilter_enlarge) {
    g_printerr ("capsFilter_enlarge could not be created. Exiting.\n");
    return -1;
  }

  // Reduce the previously enlarged image frame so that the final output video retains the
  // same dimension as the pipeline's input video dimension.
  nvvideoconvert_reduce = gst_element_factory_make("nvvideoconvert", "nvvideoconvert_reduce");
  if (!nvvideoconvert_reduce) {
    g_printerr ("nvvideoconvert_reduce could not be created. Exiting.\n");
    return -1;
  }
  capsFilter_reduce = gst_element_factory_make("capsfilter", "capsFilter_reduce");
  if (!capsFilter_reduce) {
    g_printerr ("capsFilter_reduce could not be created. Exiting.\n");
    return -1;
  }

  gchar *string1 = NULL;
  asprintf (&string1, "%d:%d:%d:%d", _pad_dim, _pad_dim, _image_width, _image_height);

  // "dest-crop" - input size < output size
  g_object_set(G_OBJECT(nvvideoconvert_enlarge), "dest-crop", string1,"interpolation-method",1 ,NULL);
  // "src-crop" - input size > output size
  g_object_set(G_OBJECT(nvvideoconvert_reduce), "src-crop", string1,"interpolation-method",1 ,NULL);
  free(string1);

  /* Padding due to AR SDK model requires bigger bboxes*/
  const int muxer_output_width_pad = _pad_dim * 2 + _image_width;
  const int muxer_output_height_pad = _pad_dim * 2 + _image_height;
  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      muxer_output_width_pad, muxer_output_height_pad);
  GstCaps *caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_enlarge),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);

  asprintf (&string1, "video/x-raw(memory:NVMM),width=%d,height=%d",
      _image_width, _image_height);
  caps1 = gst_caps_from_string (string1);
  g_object_set(G_OBJECT(capsFilter_reduce),"caps", caps1, NULL);
  free(string1);
  gst_caps_unref(caps1);
  //---Manipulate image size so that PGIE bbox is large enough---

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if (!nvosd) {
    g_printerr ("Nvdsosd could not be created. Exiting.\n");
    return -1;
  }
  nvtile = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, NULL);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad* osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref(osd_sink_pad);

  /* Set output file location */
  int sink_type = 0;
  parse_sink_type_yaml(&sink_type, argv[1]);
  int enc_type = 0;
  parse_sink_enc_type_yaml(&enc_type, argv[1]);
  g_print("sink_type:%d, enc_type:%d\n", sink_type, enc_type);

  if(sink_type == 1) {
    sink = gst_element_factory_make("nvvideoencfilesinkbin", "nv-filesink");
    if (!sink) {
      g_printerr ("Filesink could not be created. Exiting.\n");
      return -1;
    }
    g_object_set(G_OBJECT(sink), "output-file", "out.mp4", NULL);
    g_object_set(G_OBJECT(sink), "bitrate", 4000000, NULL);
    //g_object_set(G_OBJECT(sink), "profile", 3, NULL);
    g_object_set(G_OBJECT(sink), "codec", 1, NULL);//hevc
    // g_object_set(G_OBJECT(sink), "control-rate", 0, NULL);//hevc
    g_object_set(G_OBJECT(sink), "enc-type", enc_type, NULL);
  } else if(sink_type == 2) {
    sink = gst_element_factory_make("nvrtspoutsinkbin", "nv-rtspsink");
    if (!sink) {
      g_printerr ("Filesink could not be created. Exiting.\n");
      return -1;
    }
    g_object_set(G_OBJECT(sink), "enc-type", enc_type, NULL);
  } else if(sink_type == 3) {
    if (prop.integrated) {
      sink = gst_element_factory_make("nv3dsink", "nv-sink");
    } else {
#ifdef __aarch64__
      sink = gst_element_factory_make("nv3dsink", "nv-sink");
#else
      sink = gst_element_factory_make("nveglglessink", "nv-sink");
#endif
    }
  } else {
    sink = gst_element_factory_make("fakesink", "nv-fakesink");
  }

  /* Add all elements to the pipeline */
  // streammux has been added into pipeline already.
  gst_bin_add_many(GST_BIN(pipeline),
    nvvideoconvert_enlarge, capsFilter_enlarge,
    pgie, tracker, sgie, postprocess, preprocess1, sgie1, nvtile,
    nvvidconv, nvosd, sink, nvdslogger,
    nvvideoconvert_reduce, capsFilter_reduce, NULL);

  // Link elements
  if (!gst_element_link_many(streammux,
      nvvideoconvert_enlarge, capsFilter_enlarge, pgie, tracker, sgie, postprocess, preprocess1, sgie1,
      nvdslogger, nvvideoconvert_reduce, capsFilter_reduce, nvtile, nvvidconv, nvosd,  sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  /* Set the pipeline to "playing" state */
  g_print("Now playing!\n");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  GST_DEBUG_BIN_TO_DOT_FILE((GstBin*)pipeline, GST_DEBUG_GRAPH_SHOW_ALL, "pipeline");

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  return 0;

}
