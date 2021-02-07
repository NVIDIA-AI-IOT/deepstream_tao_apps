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
#include <unistd.h>
#include <iostream>
#include "gstnvdsmeta.h"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

#define TILED_OUTPUT_WIDTH 1920
#define TILED_OUTPUT_HEIGHT 1080

static gboolean
det_bus_call (GstBus * bus, GstMessage * msg, gpointer data) {
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

static void printUsage(const char* cmd) {
    g_printerr ("\tUsage: %s -c pgie_config_file -i <H264 or JPEG filename> [-b BATCH] [-d]\n", cmd);
    g_printerr ("-h: \n\tprint help info \n");
    g_printerr ("-c: \n\tpgie config file, e.g. pgie_frcnn_tlt_config.txt  \n");
    g_printerr ("-i: \n\tH264 or JPEG input file  \n");
    g_printerr ("-b: \n\tbatch size, this will override the value of \"baitch-size\" in pgie config file  \n");
    g_printerr ("-d: \n\tenable display, otherwise dump to output H264 or JPEG file  \n");
}
int
det_main (int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *parser = NULL,
               *decoder = NULL, *streammux = NULL, *sink = NULL,
               *pgie = NULL, *nvvidconv = NULL, *nvdsosd = NULL,
               *parser1 = NULL, *nvvidconv1 = NULL, *enc = NULL,
               *tiler = NULL, *tee = NULL;

#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    gboolean isH264 = FALSE;
    gboolean useDisplay = FALSE;
    guint tiler_rows, tiler_cols;
    guint batchSize = 1;
    guint pgie_batch_size;
    guint c;
    const char* optStr = "b:c:dhi:";
    std::string pgie_config;
    std::string input_file;
    gboolean showMask = FALSE;

    while ((c = getopt(argc, argv, optStr)) != -1) {
        switch (c) {
            case 'b':
                batchSize = std::atoi(optarg);
                batchSize = batchSize == 0 ? 1:batchSize;
                break;
            case 'c':
                pgie_config.assign(optarg);
                break;
            case 'd':
                useDisplay = TRUE;
                break;
            case 'i':
                input_file.assign(optarg);
                break;
            case 'h':
            default:
                printUsage(argv[0]);
                return -1;
          }
     }

    /* Check input arguments */
    if (argc == 1) {
        printUsage(argv[0]);
        return -1;
    }

    const gchar *p_end = input_file.c_str() + strlen(input_file.c_str());
    if(!strncmp(p_end - strlen("h264"), "h264", strlen("h264"))) {
        isH264 = TRUE;
    } else if(!strncmp(p_end - strlen("jpg"), "jpg", strlen("jpg")) || !strncmp(p_end - strlen("jpeg"), "jpeg", strlen("jpeg"))) {
        isH264 = FALSE;
    } else {
        g_printerr("input file only support H264 and JPEG\n");
        return -1;
    }

    const char* use_display = std::getenv("USE_DISPLAY");
    if(use_display != NULL && std::stoi(use_display) == 1) {
        useDisplay = true;
    }

    const char* batch_size = std::getenv("BATCH_SIZE");
    if(batch_size != NULL ) {
        batchSize = std::stoi(batch_size);
        g_printerr("batch size is %d \n", batchSize);
    }

    const char* show_mask = std::getenv("SHOW_MASK");
    if(show_mask != NULL && std::stoi(show_mask) == 1) {
        showMask= TRUE;
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("ds-custom-pipeline");

    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    if(isH264 == TRUE) {
        parser = gst_element_factory_make ("h264parse", "h264-parser");
    } else {
        parser = gst_element_factory_make ("jpegparse", "jpeg-parser");
    }

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvdsosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

    /* Create OSD to draw on the converted RGBA buffer */
    nvdsosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    tee = gst_element_factory_make("tee", "tee");
    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    if(useDisplay == FALSE) {
        if(isH264 == TRUE){
            parser1 = gst_element_factory_make ("h264parse", "h264-parser1");
            enc = gst_element_factory_make ("nvv4l2h264enc", "h264-enc");
        } else{
            parser1 = gst_element_factory_make ("jpegparse", "jpeg-parser1");
            enc = gst_element_factory_make ("jpegenc", "jpeg-enc");
        }
        nvvidconv1 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
        sink = gst_element_factory_make ("filesink", "file-sink");
        if (!source || !parser || !parser1 || !decoder || !tee || !pgie
                || !tiler || !nvvidconv || !nvvidconv1 || !nvdsosd || !enc || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    } else {
        sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
        if (!source || !parser || !decoder || !tee || !pgie
                || !tiler || !nvvidconv || !nvdsosd || !sink) {
            g_printerr ("One element could not be created. Exiting.\n");
            return -1;
        }
    }

#ifdef PLATFORM_TEGRA
    if(!transform) {
        g_printerr ("One tegra element could not be created. Exiting.\n");
        return -1;
    }
#endif

    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", input_file.c_str(), NULL);

    //save the file to local dir
    if(useDisplay == FALSE) {
        if(isH264 == TRUE)
            g_object_set (G_OBJECT (sink), "location", "./out.h264", NULL);
        else
            g_object_set (G_OBJECT (sink), "location", "./out.jpg", NULL);
    }
    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT, "batch-size", batchSize,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie),
                  "config-file-path", pgie_config.c_str(), NULL);

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
    if(showMask)
        g_object_set (G_OBJECT (nvdsosd), "display-mask", 1, "display-bbox", 0, "display-text", 0, "process-mode", 0, NULL);
    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, det_bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
    if(useDisplay == FALSE) {
        gst_bin_add_many (GST_BIN (pipeline),
                          source, parser, decoder, tee, streammux, pgie, tiler,
                          nvvidconv, nvdsosd, nvvidconv1, enc, parser1, sink, NULL);
    } else {
#ifdef PLATFORM_TEGRA
        gst_bin_add_many (GST_BIN (pipeline),
                          source, parser, decoder, tee, streammux, pgie,
                          tiler, nvvidconv, nvdsosd, transform, sink, NULL);
#else
        gst_bin_add_many (GST_BIN (pipeline),
                          source, parser, decoder, tee, streammux, pgie,
                          tiler, nvvidconv, nvdsosd, sink, NULL);
#endif
    }

    for(guint i = 0; i < batchSize; i++) {
        GstPad *sinkpad, *srcpad;
        gchar pad_name_sink[16] = {};
        gchar pad_name_src[16] = {};

        g_snprintf (pad_name_sink, 15, "sink_%u", i);
        g_snprintf (pad_name_src, 15, "src_%u", i);
        sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
        if (!sinkpad) {
            g_printerr ("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_request_pad(tee, pad_name_src);
        if (!srcpad) {
            g_printerr ("tee request src pad failed. Exiting.\n");
            return -1;
        }

        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr ("Failed to link tee to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref (sinkpad);
        gst_object_unref (srcpad);
    }
    /* We link the elements together */
    /* file-source -> h264-parser -> nvv4l2decoder ->
     * nvinfer -> nvvideoconvert -> nvdsosd -> video-renderer */

    if (!gst_element_link_many (source, parser, decoder, tee, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }
    if (useDisplay == FALSE) {
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, nvvidconv1, enc, parser1, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
    } else {
#ifdef PLATFORM_TEGRA
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, transform, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
#else
        if (!gst_element_link_many (streammux, pgie, tiler,
                                    nvvidconv, nvdsosd, sink, NULL)) {
            g_printerr ("Elements could not be linked: 2. Exiting.\n");
            return -1;
        }
#endif
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
