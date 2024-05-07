/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "nvds_yml_parser.h"
#include "ds_yml_parse.h"
#include <yaml-cpp/yaml.h>
#include <math.h>
#include "cuda_runtime_api.h"

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

int main (int argc, char *argv[]) {
  GMainLoop *loop = NULL;
  GstElement *streammux = NULL, *sink = NULL, 
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL, *nvtile = NULL,
             *nvocdr_template = NULL, *mux = NULL, *encparse = NULL;
  DsSourceBinStruct source_struct[128];

  GstBus *bus = NULL;
  guint bus_watch_id;
  GstCaps *caps = NULL;

  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isImage=false;
  bool isStreaming=false;
  GList* g_list = NULL;
  GList* iterator = NULL;
  bool isH264 = true;
  int enc_type = ENCODER_TYPE_HW;
  gchar *filepath = NULL;
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);


  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  /* setup singal handler */
  signal_catch_setup();
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("nvocdr-pipeline"); 

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One main element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN(pipeline), streammux);

  if (NVDS_YAML_PARSER_SUCCESS != nvds_parse_source_list(&g_list, argv[1], "source-list")) {
    g_printerr ("No source is found. Exiting.\n");
    return -1;
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

  nvocdr_template = gst_element_factory_make ("nvdsvideotemplate",
                       "nvocdr_template");
  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvid-converter");
  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvid-converter1");
  capfilt = gst_element_factory_make ("capsfilter", "nvvideo-caps");
  nvtile = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  guint output_type = 0;

  output_type = ds_parse_group_type(argv[1], "output");
  if(!output_type){
    g_printerr ("No output setting. Exiting.\n");
    return -1;
  }

  if (output_type == 1) {
    GString * filename = NULL;
    filename = ds_parse_file_name(argv[1], "output");

    if (isImage) {
        outenc = gst_element_factory_make ("jpegenc", "jpegenc");
        caps =
            gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
            "I420", NULL);
        g_object_set (G_OBJECT (capfilt), "caps", caps, NULL);
        filepath = g_strconcat(filename->str,".jpg",NULL);
    } else {
        mux = gst_element_factory_make ("qtmux", "mp4-mux");
        isH264 = !(ds_parse_enc_codec(argv[1], "output"));
        enc_type = ds_parse_enc_type(argv[1], "output");

        create_video_encoder(isH264, enc_type, &capfilt, &outenc, &encparse, NULL);
        filepath = g_strconcat(filename->str,".mp4",NULL);
        if(enc_type == ENCODER_TYPE_HW)
          ds_parse_enc_config (outenc, argv[1], "output");
    }
    sink = gst_element_factory_make ("filesink", "nvvideo-renderer");
  }
  else if (output_type == 2)
    sink = gst_element_factory_make ("fakesink", "fake-renderer");
  else if (output_type == 3) {
    if(prop.integrated)
      sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
    else
#ifdef __aarch64__
      sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");
#else
      sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!nvvidconv || !nvocdr_template || !nvosd || !sink  || !capfilt) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  nvds_parse_streammux(streammux, argv[1], "streammux");

  if (isStreaming)
    g_object_set (G_OBJECT (streammux), "live-source", true, NULL);
  g_object_set (G_OBJECT (streammux), "batch-size", src_cnt, NULL);

  tiler_rows = (guint) sqrt (src_cnt);
  tiler_columns = (guint) ceil (1.0 * src_cnt / tiler_rows);
  g_object_set (G_OBJECT (nvtile), "rows", tiler_rows, "columns",
      tiler_columns, "width", 1280, "height", 720, NULL);

  ds_parse_ocdr_videotemplate_config(nvocdr_template, argv[1], "video-template");

  /* we add a bus message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);


  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), 
    nvocdr_template, nvvidconv, nvosd, nvtile, sink, NULL);

  if (!gst_element_link_many (streammux, nvocdr_template, nvtile, nvvidconv, nvosd, NULL)) {
    g_printerr ("Inferring and tracking elements link failure.\n");
    return -1;
  }

  g_object_set (G_OBJECT (sink), "sync", 0, "async", false,NULL);

  if (output_type == 1) {
    g_object_set (G_OBJECT (sink), "location", filepath, NULL);
    g_object_set (G_OBJECT (sink), "enable-last-sample", false,NULL);
    if (!isImage) {
      gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
          encparse, mux, NULL);
      if (!gst_element_link_many (nvosd, nvvidconv1, capfilt,
             outenc, encparse, mux, sink, NULL)) {
        g_printerr ("OSD and sink elements link failure.\n");
        return -1;
      }
    } else {
      gst_bin_add_many (GST_BIN (pipeline), nvvidconv1, outenc, capfilt,
          NULL);
      if (!gst_element_link_many (nvosd, nvvidconv1, capfilt,
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
    if (!gst_element_link_many (nvosd, sink, NULL)) {
      g_printerr ("OSD and sink elements link failure.\n");
      return -1;
    }
  }

  /* Set the pipeline to "playing" state */
  g_print (" Now playing! \n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}
