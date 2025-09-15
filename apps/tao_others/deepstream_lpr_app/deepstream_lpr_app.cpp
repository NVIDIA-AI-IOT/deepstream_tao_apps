/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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
#include <glib.h>
#include <gmodule.h>
#include <gst/gst.h>
#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>

#include "ds_yml_parse.h"
#include "gst-nvmessage.h"
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "nvdsmeta.h"

#define MAX_DISPLAY_LEN 64

#define MEASURE_ENABLE 1

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define SGIE_CLASS_ID_LPD 0

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

/* Check for parsing error. */
#define RETURN_ON_PARSER_ERROR(parse_expr)                                     \
  if (NVDS_YAML_PARSER_SUCCESS != parse_expr) {                                \
    g_printerr("Error in parsing configuration file.\n");                      \
    return -1;                                                                 \
  }

gint frame_number = 0;
gint total_plate_number = 0;
gchar pgie_classes_str[4][32] = {"Vehicle", "TwoWheeler", "Person", "Roadsign"};

extern "C" void parse_nvdsanalytics_meta_data(NvDsBatchMeta *batch_meta);

#define PRIMARY_DETECTOR_UID 1
#define SECONDARY_DETECTOR_UID 2
#define SECONDARY_CLASSIFIER_UID 3

typedef struct _perf_measure {
  GstClockTime pre_time;
  GstClockTime total_time;
  guint count;
} perf_measure;

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  guint lp_count = 0;
  guint label_i = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsMetaList *l_class = NULL;
  NvDsMetaList *l_label = NULL;
  NvDsDisplayMeta *display_meta = NULL;
  NvDsClassifierMeta *class_meta = NULL;
  NvDsLabelInfo *label_info = NULL;
  GstClockTime now;
  perf_measure *perf = (perf_measure *)(u_data);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

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
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
    int offset = 0;
    if (!frame_meta)
      continue;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *)(l_obj->data);

      if (!obj_meta)
        continue;

      /* Check that the object has been detected by the primary detector
       * and that the class id is that of vehicles/persons. */
      if (obj_meta->unique_component_id == PRIMARY_DETECTOR_UID) {
        if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE)
          vehicle_count++;
        if (obj_meta->class_id == PGIE_CLASS_ID_PERSON)
          person_count++;
      }

      if (obj_meta->unique_component_id == SECONDARY_DETECTOR_UID) {
        if (obj_meta->class_id == SGIE_CLASS_ID_LPD) {
          lp_count++;
          /* Print this info only when operating in secondary model. */
          if (obj_meta->parent)
            g_print("License plate found for parent object %p (type=%s)\n",
                    obj_meta->parent,
                    pgie_classes_str[obj_meta->parent->class_id]);

          obj_meta->text_params.set_bg_clr = 1;
          obj_meta->text_params.text_bg_clr.red = 0.0;
          obj_meta->text_params.text_bg_clr.green = 0.0;
          obj_meta->text_params.text_bg_clr.blue = 0.0;
          obj_meta->text_params.text_bg_clr.alpha = 0.0;

          obj_meta->text_params.font_params.font_color.red = 1.0;
          obj_meta->text_params.font_params.font_color.green = 1.0;
          obj_meta->text_params.font_params.font_color.blue = 0.0;
          obj_meta->text_params.font_params.font_color.alpha = 1.0;
          obj_meta->text_params.font_params.font_size = 12;
        }
      }

      for (l_class = obj_meta->classifier_meta_list; l_class != NULL;
           l_class = l_class->next) {
        class_meta = (NvDsClassifierMeta *)(l_class->data);
        if (!class_meta)
          continue;
        if (class_meta->unique_component_id == SECONDARY_CLASSIFIER_UID) {
          for (label_i = 0, l_label = class_meta->label_info_list;
               label_i < class_meta->num_labels && l_label;
               label_i++, l_label = l_label->next) {
            label_info = (NvDsLabelInfo *)(l_label->data);
            if (label_info) {
              if (label_info->label_id == 0 &&
                  label_info->result_class_id == 1) {
                g_print("Plate License %s\n", label_info->result_label);
              }
            }
          }
        }
      }
    }

    display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (char *)g_malloc0(MAX_DISPLAY_LEN);
    offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
                      person_count);
    offset += snprintf(txt_params->display_text + offset, MAX_DISPLAY_LEN,
                       "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    char font_n[6];
    snprintf(font_n, 6, "Serif");
    txt_params->font_params.font_name = font_n;
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame(frame_meta, display_meta);
  }

  g_print("Frame Number = %d Vehicle Count = %d Person Count = %d"
          " License Plate Count = %d\n",
          frame_number, vehicle_count, person_count, lp_count);
  frame_number++;
  total_plate_number += lp_count;
  return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;
  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR: {
    gchar *debug;
    GError *error;
    gst_message_parse_error(msg, &error, &debug);
    g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src),
               error->message);
    if (debug)
      g_printerr("Error details: %s\n", debug);
    g_free(debug);
    g_error_free(error);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }
  return TRUE;
}

static void cb_new_pad(GstElement *element, GstPad *pad, GstElement *data) {
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  GstPadLinkReturn ret;

  GstPad *sink_pad = gst_element_get_static_pad(data, "sink");
  if (gst_pad_is_linked(sink_pad)) {
    g_print("h264parser already linked. Ignoring.\n");
    goto exit;
  }

  new_pad_caps = gst_pad_get_current_caps(pad);
  new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
  new_pad_type = gst_structure_get_name(new_pad_struct);
  g_print("qtdemux pad %s\n", new_pad_type);

  if (g_str_has_prefix(new_pad_type, "video/x-h264")) {
    ret = gst_pad_link(pad, sink_pad);
    if (GST_PAD_LINK_FAILED(ret))
      g_print("fail to link parser and mp4 demux.\n");
  } else {
    g_print("%s output, not 264 stream\n", new_pad_type);
  }

exit:
  gst_object_unref(sink_pad);
}

/* nvdsanalytics_src_pad_buffer_probe  will extract metadata received on
 * nvdsanalytics src pad and extract nvanalytics metadata etc. */
static GstPadProbeReturn
nvdsanalytics_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info,
                                   gpointer u_data) {
  GstBuffer *buf = (GstBuffer *)info->data;
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

  parse_nvdsanalytics_meta_data(batch_meta);

  return GST_PAD_PROBE_OK;
}

static inline const char *infer_plugin(NvDsGieType type) {
  switch (type) {
  case NVDS_GIE_PLUGIN_INFER:
    return "nvinfer";
  case NVDS_GIE_PLUGIN_INFER_SERVER:
    return "nvinferserver";
  default:
    return "unknown";
  }
}

int main(int argc, char *argv[]) {
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL,
             *primary_detector = NULL, *secondary_detector = NULL,
             *nvvidconv = NULL, *nvosd = NULL, *nvvidconv1 = NULL,
             *outenc = NULL, *capfilt = NULL, *mux = NULL,
             *secondary_classifier = NULL, *nvtile = NULL, *encparse = NULL;
  GstElement *tracker = NULL, *nvdsanalytics = NULL;
  GstElement *queue1 = NULL, *queue2 = NULL, *queue3 = NULL, *queue4 = NULL,
             *queue5 = NULL, *queue6 = NULL, *queue7 = NULL, *queue8 = NULL,
             *queue9 = NULL, *queue10 = NULL;
  GstElement *h264parser[128], *source[128], *decoder[128], *mp4demux[128],
      *parsequeue[128];
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;
  // int i;
  static guint src_cnt = 0;
  guint tiler_rows, tiler_columns;
  perf_measure perf_measure;

  gchar ele_name[64];
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  bool isH264 = true;
  int enc_type = ENCODER_TYPE_HW;
  GList *g_list = NULL;
  GList *iterator = NULL;

  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie0_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;

  if (argc != 2) {
    g_printerr("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  // For Chinese language supporting
  setlocale(LC_CTYPE, "");
  /* Standard GStreamer initialization */
  gst_init(&argc, &argv);
  loop = g_main_loop_new(NULL, FALSE);

  perf_measure.pre_time = GST_CLOCK_TIME_NONE;
  perf_measure.total_time = GST_CLOCK_TIME_NONE;
  perf_measure.count = 0;

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new("pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add(GST_BIN(pipeline), streammux);

  RETURN_ON_PARSER_ERROR(
      nvds_parse_source_list(&g_list, argv[1], "source-list"));
  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie"));
  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&sgie0_type, argv[1], "secondary-gie0"));
  RETURN_ON_PARSER_ERROR(
      nvds_parse_gie_type(&sgie1_type, argv[1], "secondary-gie1"));

  /* Multiple source files */
  for (iterator = g_list, src_cnt = 0; iterator;
       iterator = iterator->next, src_cnt++) {
    /* Only h264 element stream with mp4 container is supported. */
    g_snprintf(ele_name, 64, "file_src_%d", src_cnt);

    /* Source element for reading from the file */
    source[src_cnt] = gst_element_factory_make("filesrc", ele_name);

    g_snprintf(ele_name, 64, "mp4demux_%d", src_cnt);
    mp4demux[src_cnt] = gst_element_factory_make("qtdemux", ele_name);

    g_snprintf(ele_name, 64, "h264parser_%d", src_cnt);
    h264parser[src_cnt] = gst_element_factory_make("h264parse", ele_name);

    g_snprintf(ele_name, 64, "parsequeue_%d", src_cnt);
    parsequeue[src_cnt] = gst_element_factory_make("queue", ele_name);

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    g_snprintf(ele_name, 64, "decoder_%d", src_cnt);
    decoder[src_cnt] = gst_element_factory_make("nvv4l2decoder", ele_name);

    if (!source[src_cnt] || !h264parser[src_cnt] || !decoder[src_cnt] ||
        !mp4demux[src_cnt]) {
      g_printerr("One element could not be created. Exiting.\n");
      return -1;
    }

    gst_bin_add_many(GST_BIN(pipeline), source[src_cnt], mp4demux[src_cnt],
                     h264parser[src_cnt], parsequeue[src_cnt], decoder[src_cnt],
                     NULL);

    g_snprintf(pad_name_sink, 64, "sink_%d", src_cnt);
    sinkpad = gst_element_request_pad_simple(streammux, pad_name_sink);
    g_print("Request %s pad from streammux\n", pad_name_sink);
    if (!sinkpad) {
      g_printerr("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(decoder[src_cnt], pad_name_src);
    if (!srcpad) {
      g_printerr("Decoder request src pad failed. Exiting.\n");
      return -1;
    }

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
    }

    if (!gst_element_link_pads(source[src_cnt], "src", mp4demux[src_cnt],
                               "sink")) {
      g_printerr("Elements could not be linked: 0. Exiting.\n");
      return -1;
    }

    g_signal_connect(mp4demux[src_cnt], "pad-added", G_CALLBACK(cb_new_pad),
                     h264parser[src_cnt]);

    if (!gst_element_link_many(h264parser[src_cnt], parsequeue[src_cnt],
                               decoder[src_cnt], NULL)) {
      g_printerr("Elements could not be linked: 1. Exiting.\n");
    }

    /* we set the input filename to the source element */
    g_object_set(G_OBJECT(source[src_cnt]), "location", (gchar *)iterator->data,
                 NULL);

    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);
  }
  g_list_free(g_list);

  /* Create three nvinfer instances for two detectors and one classifier*/
  primary_detector = gst_element_factory_make(infer_plugin(pgie_type),
                                              "primary-infer-engine1");

  secondary_detector = gst_element_factory_make(infer_plugin(sgie0_type),
                                                "secondary-infer-engine1");

  secondary_classifier = gst_element_factory_make(infer_plugin(sgie1_type),
                                                  "secondary-infer-engine2");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvid-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

  nvvidconv1 = gst_element_factory_make("nvvideoconvert", "nvvid-converter1");

  capfilt = gst_element_factory_make("capsfilter", "nvvideo-caps");

  nvtile = gst_element_factory_make("nvmultistreamtiler", "nvtiler");

  tracker = gst_element_factory_make("nvtracker", "nvtracker");

  /* Use nvdsanalytics to perform analytics on object */
  nvdsanalytics = gst_element_factory_make("nvdsanalytics", "nvdsanalytics");

  queue1 = gst_element_factory_make("queue", "queue1");
  queue2 = gst_element_factory_make("queue", "queue2");
  queue3 = gst_element_factory_make("queue", "queue3");
  queue4 = gst_element_factory_make("queue", "queue4");
  queue5 = gst_element_factory_make("queue", "queue5");
  queue6 = gst_element_factory_make("queue", "queue6");
  queue7 = gst_element_factory_make("queue", "queue7");
  queue8 = gst_element_factory_make("queue", "queue8");
  queue9 = gst_element_factory_make("queue", "queue9");
  queue10 = gst_element_factory_make("queue", "queue10");

  /* set properties for nvdsanalytics */
  ds_parse_nvdsanalytics(nvdsanalytics, argv[1], "analytics");

  guint output_type = 2;
  output_type = ds_parse_group_type(argv[1], "output");
  if (output_type == 1) {
    sink = gst_element_factory_make("filesink", "nvvideo-renderer");
  } else if (output_type == 2) {
    sink = gst_element_factory_make("fakesink", "fake-renderer");
  } else if (output_type == 3) {
    if (prop.integrated)
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
    else
#ifdef __aarch64__
      sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
#else
      sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
#endif
  }

  if (!primary_detector || !secondary_detector || !nvvidconv || !nvosd ||
      !sink) {
    g_printerr("One element could not be created. Exiting.\n");
    return -1;
  }

  g_object_set(G_OBJECT(streammux), "width", MUXER_OUTPUT_WIDTH, "height",
               MUXER_OUTPUT_HEIGHT, "batch-size", src_cnt,
               "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  tiler_rows = (guint)sqrt(src_cnt);
  tiler_columns = (guint)ceil(1.0 * src_cnt / tiler_rows);
  g_object_set(G_OBJECT(nvtile), "rows", tiler_rows, "columns", tiler_columns,
               "width", 1280, "height", 720, NULL);

  /* Set the config files for the two detectors and one classifier. The PGIE
   * detects the cars. The first SGIE detects car plates from the cars and the
   * second SGIE classifies the caracters in the car plate to identify the car
   * plate string. */
  nvds_parse_gie(primary_detector, argv[1], "primary-gie");
  nvds_parse_gie(secondary_detector, argv[1], "secondary-gie0");
  nvds_parse_gie(secondary_classifier, argv[1], "secondary-gie1");
  nvds_parse_tracker(tracker, argv[1], "tracker");

  /* we add a bus message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many(GST_BIN(pipeline), primary_detector, secondary_detector,
                   tracker, nvdsanalytics, queue1, queue2, queue3, queue4,
                   queue5, queue6, queue7, queue8, secondary_classifier,
                   nvvidconv, nvosd, nvtile, sink, NULL);

  if (!gst_element_link_many(streammux, queue1, primary_detector, queue2,
                             tracker, queue3, nvdsanalytics, queue4,
                             secondary_detector, queue5, secondary_classifier,
                             queue6, nvtile, queue7, nvvidconv, queue8, nvosd,
                             NULL)) {
    g_printerr("Inferring and tracking elements link failure.\n");
    return -1;
  }

  if (output_type == 1) {
    isH264 = !(ds_parse_enc_codec(argv[1], "output"));
    enc_type = ds_parse_enc_type(argv[1], "output");
    create_video_encoder(isH264, enc_type, &capfilt, &outenc, &encparse, NULL);
    if (!capfilt || !outenc || !encparse) {
      g_printerr("enc element could not be created. Exiting.\n");
      return -1;
    }
    gchar *filepath = NULL;
    mux = gst_element_factory_make("qtmux", "mp4-mux");

    GString *output_file = ds_parse_file_name(argv[1], "output");
    filepath = g_strconcat(output_file->str, ".mp4", NULL);
    ds_parse_enc_config(outenc, argv[1], "output");

    g_object_set(G_OBJECT(sink), "async", FALSE, NULL);
    g_object_set(G_OBJECT(sink), "sync", TRUE, NULL);
    g_object_set(G_OBJECT(sink), "location", filepath, NULL);
    gst_bin_add_many(GST_BIN(pipeline), queue9, nvvidconv1, capfilt, queue10,
                     outenc, encparse, mux, sink, NULL);

    if (!gst_element_link_many(nvosd, queue9, nvvidconv1, capfilt, queue10,
                               outenc, encparse, mux, sink, NULL)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 2) {
    g_object_set(G_OBJECT(sink), "sync", 0, "async", false, NULL);
    if (!gst_element_link(nvosd, sink)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  } else if (output_type == 3) {
    gst_bin_add(GST_BIN(pipeline), queue9);
    if (!gst_element_link_many(nvosd, queue9, sink, NULL)) {
      g_printerr("OSD and sink elements link failure.\n");
      return -1;
    }
  }

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
  if (!osd_sink_pad)
    g_print("Unable to get sink pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      osd_sink_pad_buffer_probe, &perf_measure, NULL);
  gst_object_unref(osd_sink_pad);

  osd_sink_pad = gst_element_get_static_pad(nvdsanalytics, "src");
  if (!osd_sink_pad)
    g_print("Unable to get src pad\n");
  else
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                      nvdsanalytics_src_pad_buffer_probe, NULL, NULL);
  gst_object_unref(osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print("Now playing: %s\n", argv[1]);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print("Running...\n");
  g_main_loop_run(loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);

  g_print("Average fps %f\n", ((perf_measure.count - 1) * src_cnt * 1000000.0) /
                                  perf_measure.total_time);
  g_print("Totally %d plates are inferred\n", total_plate_number);
  g_print("Deleting pipeline\n");
  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);
  return 0;
}
