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
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "gstnvdsmeta.h"
#include "gst-nvmessage.h"

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
#define MODEL_OUTPUT_WIDTH 960
#define MODEL_OUTPUT_HEIGHT 608

#define NVINFER_PLUGIN "nvinfer"


/* tiler_sink_pad_buffer_probe  will extract metadata received on segmentation
 *  src pad */
static GstPadProbeReturn
tiler_src_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsMetaList * l_frame = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        // TODO:
    }
    return GST_PAD_PROBE_OK;
}

static gboolean
seg_bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      // Add the delay to show the result
      usleep(2000000);
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
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
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  GstElement *source, *jpegparser, *decoder;

  source = gst_element_factory_make ("filesrc", "source");

  jpegparser = gst_element_factory_make ("jpegparse", "jpeg-parser");

  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  if (!source || !jpegparser || !decoder)
  {
    g_printerr ("One element could not be created. Exiting.\n");
    return NULL;
  }
  g_object_set (G_OBJECT (source), "location", uri, NULL);
  const char *dot = strrchr(uri, '.');
  if ((!strcmp (dot+1, "mjpeg")) || (!strcmp (dot+1, "mjpg")))
  {
#ifdef PLATFORM_TEGRA
    g_object_set (G_OBJECT (decoder), "mjpeg", 1, NULL);
#endif
  }

  gst_bin_add_many (GST_BIN (bin), source, jpegparser, decoder, NULL);

  gst_element_link_many (source, jpegparser, decoder, NULL);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  GstPad *srcpad = gst_element_get_static_pad (decoder, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return NULL;
  }
  GstPad *bin_ghost_pad = gst_element_get_static_pad (bin, "src");
  if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
        srcpad)) {
    g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
  }

  return bin;
}

static void
usage(const char *bin)
{
  g_printerr ("Usage: %s  config_file <file1> [file2] ... [fileN] \n", bin);
}

int
seg_main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *seg = NULL,
             *nvsegvisual = NULL, *tiler = NULL;
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *seg_src_pad = NULL;
  guint i, num_sources = 0;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  GList *files = NULL;
  gboolean is_nvinfer_server = FALSE;
  gchar *infer_config_file = NULL;

  /* Check input arguments */
  if (argc < 3) {
    usage(argv[0]);
    return -1;
  }

  num_sources = argc - 2;

  if (num_sources == 0) {
    usage(argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest-image-decode-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i+2]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to infer on batched frame. */
  seg = gst_element_factory_make (
           NVINFER_PLUGIN,
          "primary-nvinference-engine");

  nvsegvisual = gst_element_factory_make ("nvsegvisual", "nvsegvisual");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "transform");
#endif

  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!seg || !nvsegvisual || !tiler || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

#ifdef PLATFORM_TEGRA
  if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  g_object_set (G_OBJECT (seg), "config-file-path", argv[1], NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (seg), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources && !is_nvinfer_server) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (seg), "batch-size", num_sources, NULL);
  }

  g_object_set (G_OBJECT (nvsegvisual), "batch-size", num_sources, NULL);
  g_object_set (G_OBJECT (nvsegvisual), "width", MODEL_OUTPUT_WIDTH, NULL);
  g_object_set (G_OBJECT (nvsegvisual), "height", MODEL_OUTPUT_HEIGHT, NULL);

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set(G_OBJECT(sink), "async", FALSE, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus,seg_bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* Add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), seg, nvsegvisual, tiler, transform, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvsegvidsual -> nvtiler -> transform -> video-renderer */
  if (!gst_element_link_many (streammux, seg, nvsegvisual, tiler, transform, sink, NULL))
  {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
  gst_bin_add_many (GST_BIN (pipeline), seg, nvsegvisual, tiler, sink, NULL);
  /* Link the elements together
   * nvstreammux -> nvinfer -> nvsegvisual -> nvtiler -> video-renderer */
  if (!gst_element_link_many (streammux, seg, nvsegvisual, tiler, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the src pad of the nvseg element, since by that time, the buffer would have
   * had got all the segmentation metadata. */
  seg_src_pad = gst_element_get_static_pad (seg, "src");
  if (!seg_src_pad)
    g_print ("Unable to get src pad\n");
  else
    gst_pad_add_probe (seg_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        tiler_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref (seg_src_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i+2]);
  }
  g_print ("\n");
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
