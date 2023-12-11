/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <sys/inotify.h>

#include <time.h>
#include <unistd.h>
#include <errno.h>

#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"
#include "nvds_version.h"

#include <termios.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include <cuda_runtime_api.h>
#include "deepstream_mdx_perception_app.h"

#define MAX_DISPLAY_LEN (64)
#define MAX_TIME_STAMP_LEN (64)
#define STREAMMUX_BUFFER_POOL_SIZE (16)

#define INOTIFY_EVENT_SIZE    (sizeof (struct inotify_event))
#define INOTIFY_EVENT_BUF_LEN (1024 * ( INOTIFY_EVENT_SIZE + 16))

#define IS_YAML(file) (g_str_has_suffix(file, ".yml") || g_str_has_suffix(file, ".yaml"))
#define NVDS_OBJECT_USER_EMBEDDING (nvds_get_user_meta_type("NVIDIA.NVINFER.EMBEDDING"))
/** @{
 * Macro's below and corresponding code-blocks are used to demonstrate
 * nvmsgconv + Broker Metadata manipulation possibility
 */

/**
 * IMPORTANT Note 1:
 * The code within the check for model_used == APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE
 * is applicable as sample demo code for
 * configs that use resnet PGIE model
 * with class ID's: {0, 1, 2, 3} for {CAR, BICYCLE, PERSON, ROADSIGN}
 * followed by optional Tracker + 3 X SGIEs (Vehicle-Type,Color,Make)
 * only!
 * Please comment out the code if using any other
 * custom PGIE + SGIE combinations
 * and use the code as reference to write your own
 * NvDsEventMsgMeta generation code in generate_event_msg_meta()
 * function
 */
typedef enum
{
  APP_CONFIG_ANALYTICS_MODELS_UNKNOWN = 0,
  APP_CONFIG_ANALYTICS_RETAIL = 1,
  APP_CONFIG_ANALYTICS_PEOPLE_REID = 2,
  APP_CONFIG_ANALYTICS_RETAIL_DUAL_HEAD_SGIE = 3,
} AppConfigAnalyticsModel;

#define MAX_LINES_IN_LABEL_FILE 320
#define MAX_CHAR_LENGTH_PER_LINE 100

// Variable to store labels for FSL dual head model
char labels[MAX_LINES_IN_LABEL_FILE][MAX_CHAR_LENGTH_PER_LINE];
guint dual_head_classes = 315;

/**
 * IMPORTANT Note 2:
 * GENERATE_DUMMY_META_EXT macro implements code
 * that assumes APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE
 * case discussed above, and generate dummy metadata
 * for other classes like Person class
 *
 * Vehicle class schema meta (NvDsVehicleObject) is filled
 * in properly from Classifier-Metadata;
 * see in-code documentation and usage of
 * schema_fill_sample_sgie_vehicle_metadata()
 */
#define GENERATE_DUMMY_META_EXT

/** Following class-ID's
 * used for demonstration code
 * assume an ITS detection model
 * which outputs CLASS_ID=0 for Vehicle class
 * and CLASS_ID=2 for Person class
 * and SGIEs X 3 same as the sample DS config for test5-app:
 * configs/test5_config_file_src_infer_tracker_sgie.txt
 */

#define SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID  (4)
#define SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID (5)
#define SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID  (6)
#define SECONDARY_GIE_MTMC_PERSON (0)

#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_CAR    (0)
#ifdef GENERATE_DUMMY_META_EXT
#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_PERSON (2)
#endif
/** @} */

#ifdef EN_DEBUG
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

static TestAppCtx *testAppCtx;
GST_DEBUG_CATEGORY (NVDS_APP);

/** @{ imported from deepstream-app as is */


#define MAX_INSTANCES 128
#define APP_TITLE "DeepStreamMDXPerception"

#define DEFAULT_X_WINDOW_WIDTH 1920
#define DEFAULT_X_WINDOW_HEIGHT 1080

AppCtx *appCtx[MAX_INSTANCES];
static guint cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar **cfg_files = NULL;
static gchar **input_files = NULL;
static gchar **override_cfg_file = NULL;
static gboolean playback_utc = FALSE;
static gboolean print_version = FALSE;
static gboolean show_bbox_text = FALSE;
static gboolean force_tcp = TRUE;
static gboolean print_dependencies_version = FALSE;
static gboolean quit = FALSE;
static gint return_value = 0;
static guint num_instances;
static guint num_input_files;
static GMutex fps_lock;
static gdouble fps[MAX_SOURCE_BINS];
static gdouble fps_avg[MAX_SOURCE_BINS];

static Display *display = NULL;
static Window windows[MAX_INSTANCES] = { 0 };

static GThread *x_event_thread = NULL;
static GMutex disp_lock;

static guint rrow, rcol, rcfg;
static gboolean rrowsel = FALSE, selecting = FALSE;
static AppConfigAnalyticsModel model_used = APP_CONFIG_ANALYTICS_MODELS_UNKNOWN;

static struct timeval ota_request_time;
static struct timeval ota_completion_time;

typedef struct _OTAInfo
{
  AppCtx *appCtx;
  gchar *override_cfg_file;
} OTAInfo;

/** @} imported from deepstream-app as is */
GOptionEntry entries[] = {
  {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
      "Print DeepStreamSDK version", NULL}
  ,
  {"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
      "Display Bounding box labels in tiled mode", NULL}
  ,
  {"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
      "Print DeepStreamSDK and dependencies version", NULL}
  ,
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
      "Set the config file", NULL}
  ,
  {"override-cfg-file", 'o', 0, G_OPTION_ARG_FILENAME_ARRAY, &override_cfg_file,
      "Set the override config file, used for on-the-fly model update feature",
        NULL}
  ,
  {"input-file", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_files,
      "Set the input file", NULL}
  ,
  {"playback-utc", 'p', 0, G_OPTION_ARG_INT, &playback_utc,
        "Playback utc; default=false (base UTC from file-URL or RTCP Sender Report) =true (base UTC from file/rtsp URL)",
      NULL}
  ,
  {"pgie-model-used", 'm', 0, G_OPTION_ARG_INT, &model_used,
        "PGIE Model used; {0 - Unknown [DEFAULT]}, {1: FSL}, {2: MTMC}, {3: FSL Dual Head SGIE}}",
      NULL}
  ,
  {"no-force-tcp", 0, G_OPTION_FLAG_REVERSE, G_OPTION_ARG_NONE, &force_tcp,
      "Do not force TCP for RTP transport", NULL}
  ,
  {NULL}
  ,
};

/**
 * @brief  Fill NvDsVehicleObject with the NvDsClassifierMetaList
 *         information in NvDsObjectMeta
 *         NOTE: This function assumes the test-application is
 *         run with 3 X SGIEs sample config:
 *         test5_config_file_src_infer_tracker_sgie.txt
 *         or an equivalent config
 *         NOTE: If user is adding custom SGIEs, make sure to
 *         edit this function implementation
 * @param  obj_params [IN] The NvDsObjectMeta as detected and kept
 *         in NvDsBatchMeta->NvDsFrameMeta(List)->NvDsObjectMeta(List)
 * @param  obj [IN/OUT] The NvDSMeta-Schema defined Vehicle metadata
 *         structure
 */
static void schema_fill_sample_sgie_vehicle_metadata (NvDsObjectMeta *
    obj_params, NvDsVehicleObject * obj);

/**
 * @brief  Performs model update OTA operation
 *         Sets "model-engine-file" configuration parameter
 *         on infer plugin to initiate model switch OTA process
 * @param  ota_appCtx [IN] App context pointer
 */
void apply_ota (AppCtx * ota_appCtx);

/**
 * @brief  Thread which handles the model-update OTA functionlity
 *         1) Adds watch on the changes made in the provided ota-override-file,
 *            if changes are detected, validate the model-update change request,
 *            intiate model-update OTA process
 *         2) Frame drops / frames without inference should NOT be detected in
 *            this on-the-fly model update process
 *         3) In case of model update OTA fails, error message will be printed
 *            on the console and pipeline continues to run with older
 *            model configuration
 * @param  gpointer [IN] Pointer to OTAInfo structure
 * @param  gpointer [OUT] Returns NULL in case of thread exits
 */
gpointer ota_handler_thread (gpointer data);

static void generate_ts_rfc3339(char *buf, int buf_size) {
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6];  //.nnnZ\0

  clock_gettime(CLOCK_REALTIME, &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec / 1000000;
  g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static GstClockTime generate_ts_rfc3339_from_ts(char *buf, int buf_size,
                                                GstClockTime ts, gchar *src_uri,
                                                gint stream_id) {
  time_t tloc;
  struct tm tm_log;
  char strmsec[6];  //.nnnZ\0
  int ms;

  GstClockTime ts_generated;

  if (playback_utc || (appCtx[0]->config.multi_source_config[stream_id].type !=
                       NV_DS_SOURCE_RTSP)) {
    if (testAppCtx->streams[stream_id].meta_number == 0) {
      testAppCtx->streams[stream_id].timespec_first_frame =
          extract_utc_from_uri(src_uri);
      memcpy(
          &tloc,
          (void *)(&testAppCtx->streams[stream_id].timespec_first_frame.tv_sec),
          sizeof(time_t));
      ms =
          testAppCtx->streams[stream_id].timespec_first_frame.tv_nsec / 1000000;
      testAppCtx->streams[stream_id].gst_ts_first_frame = ts;
      ts_generated = GST_TIMESPEC_TO_TIME(
          testAppCtx->streams[stream_id].timespec_first_frame);
      if (ts_generated == 0) {
        // Expected warning for default stream
        // g_print(
        //     "WARNING; playback mode used with URI [%s] not conforming to "
        //     "timestamp format;"
        //     " check README; using system-time\n",
        //     src_uri);

        clock_gettime(CLOCK_REALTIME,
                      &testAppCtx->streams[stream_id].timespec_first_frame);
        ts_generated = GST_TIMESPEC_TO_TIME(
            testAppCtx->streams[stream_id].timespec_first_frame);
      }
    } else {
      GstClockTime ts_current =
          GST_TIMESPEC_TO_TIME(
              testAppCtx->streams[stream_id].timespec_first_frame) +
          (ts - testAppCtx->streams[stream_id].gst_ts_first_frame);
      struct timespec timespec_current;
      GST_TIME_TO_TIMESPEC(ts_current, timespec_current);
      memcpy(&tloc, (void *)(&timespec_current.tv_sec), sizeof(time_t));
      ms = timespec_current.tv_nsec / 1000000;
      ts_generated = ts_current;
    }
  } else {
    /** ts itself is UTC Time in ns */
    struct timespec timespec_current;
    GST_TIME_TO_TIMESPEC(ts, timespec_current);
    memcpy(&tloc, (void *)(&timespec_current.tv_sec), sizeof(time_t));
    ms = timespec_current.tv_nsec / 1000000;
    ts_generated = ts;
  }
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  g_snprintf(strmsec, sizeof(strmsec), ".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);

  return ts_generated;
}

static gpointer meta_copy_func(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *)g_memdup(srcMeta, sizeof(NvDsEventMsgMeta));

  if (srcMeta->ts) dstMeta->ts = g_strdup(srcMeta->ts);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *)g_memdup(
        srcMeta->objSignature.signature, srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup(srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    dstMeta->sensorStr = g_strdup(srcMeta->sensorStr);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *)srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender) obj->gender = g_strdup(srcObj->gender);
      if (srcObj->cap) obj->cap = g_strdup(srcObj->cap);
      if (srcObj->hair) obj->hair = g_strdup(srcObj->hair);
      if (srcObj->apparel) obj->apparel = g_strdup(srcObj->apparel);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof(NvDsPersonObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PRODUCT) {
      NvDsProductObject *srcObj = (NvDsProductObject *)srcMeta->extMsg;
      NvDsProductObject *obj =
          (NvDsProductObject *)g_malloc0(sizeof(NvDsProductObject));
      if (srcObj->brand) obj->brand = g_strdup(srcObj->brand);
      if (srcObj->type) obj->type = g_strdup(srcObj->type);
      if (srcObj->shape) obj->shape = g_strdup(srcObj->shape);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof(NvDsProductObject);
    }
  }

  if (srcMeta->embedding.embedding_length > 0) {
    dstMeta->embedding.embedding_length = srcMeta->embedding.embedding_length;
    dstMeta->embedding.embedding_vector =
        g_memdup(srcMeta->embedding.embedding_vector,
                 srcMeta->embedding.embedding_length * sizeof(float));
  }

  return dstMeta;
}

static void meta_free_func(gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *)user_meta->user_meta_data;
  user_meta->user_meta_data = NULL;

  if (srcMeta->ts) {
    g_free(srcMeta->ts);
  }

  if (srcMeta->objSignature.size > 0) {
    g_free(srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free(srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    g_free(srcMeta->sensorStr);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *)srcMeta->extMsg;

      if (obj->gender) g_free(obj->gender);
      if (obj->cap) g_free(obj->cap);
      if (obj->hair) g_free(obj->hair);
      if (obj->apparel) g_free(obj->apparel);
    }
    //! Extensions for Fewshot Learning
    else if (srcMeta->objType == NVDS_OBJECT_TYPE_PRODUCT) {
      NvDsProductObject *obj = (NvDsProductObject *)srcMeta->extMsg;

      if (obj->brand) g_free(obj->brand);
      if (obj->type) g_free(obj->type);
      if (obj->shape) g_free(obj->shape);
    }

    g_free(srcMeta->extMsg);
    srcMeta->extMsg = NULL;
    srcMeta->extMsgSize = 0;
  }

  if (srcMeta->embedding.embedding_vector) {
    g_free(srcMeta->embedding.embedding_vector);
  }
  srcMeta->embedding.embedding_length = 0;

  g_free(srcMeta);
}

#ifdef GENERATE_DUMMY_META_EXT
static void generate_person_meta(gpointer data) {
  NvDsPersonObject *obj = (NvDsPersonObject *)data;
  obj->age = 45;
  obj->cap = g_strdup("none-dummy-person-info");
  obj->hair = g_strdup("black");
  obj->gender = g_strdup("male");
  obj->apparel = g_strdup("formal");
}
//! Extensions for Fewshot Learning
// Create product meta object
static void generate_product_meta(gpointer data) {
  NvDsProductObject *obj = (NvDsProductObject *)data;
  obj->brand = g_strdup("");
  obj->type = g_strdup("");
  obj->shape = g_strdup("");
}
#endif

static void generate_event_msg_meta(AppCtx *appCtx, gpointer data,
                                    gint class_id, gboolean useTs,
                                    GstClockTime ts, gchar *src_uri,
                                    gint stream_id, guint sensor_id,
                                    NvDsObjectMeta *obj_params, float scaleW,
                                    float scaleH, NvDsFrameMeta *frame_meta,
                                    NvDsInferTensorMeta *tensor_meta,
                                    NvDsBatchMeta *batch_meta) {
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *)data;
  GstClockTime ts_generated = 0;

  meta->objType = NVDS_OBJECT_TYPE_UNKNOWN; /**< object unknown */
  /* The sensor_id is parsed from the source group name which has the format
   * [source<sensor-id>]. */
  meta->sensorId = sensor_id;
  meta->placeId = sensor_id;
  meta->moduleId = sensor_id;
  meta->frameId = frame_meta->frame_num;
  meta->ts = (gchar *)g_malloc0(MAX_TIME_STAMP_LEN + 1);
  meta->objectId = (gchar *)g_malloc0(MAX_LABEL_SIZE);
  meta->bbox.left = obj_params->rect_params.left * scaleW;
  meta->bbox.top = obj_params->rect_params.top * scaleH;
  meta->bbox.width = obj_params->rect_params.width * scaleW;
  meta->bbox.height = obj_params->rect_params.height * scaleH;
  meta->confidence = obj_params->confidence;
  /** tracking ID */
  meta->trackingId = obj_params->object_id;

  strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

  /** INFO: This API is called once for every 30 frames (now) */
  if (useTs && src_uri) {
    ts_generated = generate_ts_rfc3339_from_ts(meta->ts, MAX_TIME_STAMP_LEN, ts,
                                               src_uri, stream_id);
  } else {
    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);
  }
  /** sensor ID when streams are added using nvmultiurisrcbin REST API */
  NvDsSensorInfo *sensorInfo = get_sensor_info(appCtx, stream_id);
  if (sensorInfo) {
    /** this stream was added using REST API; we have Sensor Info! */
    // g_print(
    //     "this stream [%d:%s] was added using REST API; we have Sensor
    //     Info\n", sensorInfo->source_id, sensorInfo->sensor_id);
    meta->sensorStr = g_strdup(sensorInfo->sensor_id);
  }

  (void)ts_generated;
  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject / NvDsProductObject. Then that
   * object should be handled in gst-nvmsgconv component accordingly.
   */
  if (model_used == APP_CONFIG_ANALYTICS_RETAIL) {
    
  } else if (model_used == APP_CONFIG_ANALYTICS_PEOPLE_REID) {
    if (class_id == SECONDARY_GIE_MTMC_PERSON) {
      meta->type = NVDS_EVENT_MOVING;
      meta->objType = NVDS_OBJECT_TYPE_PERSON;
#ifdef GENERATE_DUMMY_META_EXT
      // MTMC does not require to send dummy sgie attributes
      // NvDsPersonObject *obj =
      //     (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));
      // generate_person_meta(obj);

      // meta->extMsg = obj;
      // meta->extMsgSize = sizeof(NvDsPersonObject);
#endif
    }
  }
  //
  bool embedding_on_device = TRUE;
  gchar dual_head_label[MAX_CHAR_LENGTH_PER_LINE];
  if (tensor_meta) {
    // Postprocess - embedding vector
    if (model_used == APP_CONFIG_ANALYTICS_RETAIL) {
      meta->type = NVDS_EVENT_MOVING;
      meta->objType = NVDS_OBJECT_TYPE_PRODUCT;
#ifdef GENERATE_DUMMY_META_EXT
      NvDsProductObject *obj =
          (NvDsProductObject *)g_malloc0(sizeof(NvDsProductObject));
      generate_product_meta(obj);

      meta->extMsg = obj;
      meta->extMsgSize = sizeof(NvDsProductObject);
#endif
      NvDsInferDims embedding_dims =
          tensor_meta->output_layers_info[0].inferDims;
      int numElements = embedding_dims.d[0];
      meta->embedding.embedding_vector =
          (float *)g_malloc0(numElements * sizeof(float));
      cudaMemcpy(meta->embedding.embedding_vector,
                 (float *)(tensor_meta->out_buf_ptrs_dev[0]),
                 numElements * sizeof(float), cudaMemcpyDeviceToHost);
      meta->embedding.embedding_length = numElements;
    }
    else if (model_used == APP_CONFIG_ANALYTICS_PEOPLE_REID) 
    {
      meta->type = NVDS_EVENT_MOVING;
      meta->objType = NVDS_OBJECT_TYPE_PERSON;
#ifdef GENERATE_DUMMY_META_EXT
      // MTMC does not require to send dummy sgie attributes
      // NvDsPersonObject *obj =
      //     (NvDsPersonObject *)g_malloc0(sizeof(NvDsPersonObject));
      // generate_person_meta(obj);

      // meta->extMsg = obj;
      // meta->extMsgSize = sizeof(NvDsPersonObject);
#endif
      NvDsInferDims embedding_dims =
          tensor_meta->output_layers_info[0].inferDims;
      int numElements = embedding_dims.d[0];
      meta->embedding.embedding_vector =
          (float *)g_malloc0(numElements * sizeof(float));
      cudaMemcpy(meta->embedding.embedding_vector,
                 (float *)(tensor_meta->out_buf_ptrs_dev[0]),
                 numElements * sizeof(float), cudaMemcpyDeviceToHost);
      meta->embedding.embedding_length = numElements;
    } 
    else if (model_used == APP_CONFIG_ANALYTICS_RETAIL_DUAL_HEAD_SGIE) 
    {
      meta->type = NVDS_EVENT_MOVING;
      meta->objType = NVDS_OBJECT_TYPE_PRODUCT;

      NvDsInferDims embedding_dims =
          tensor_meta->output_layers_info[0].inferDims;
      int numElements = embedding_dims.d[0];
      meta->embedding.embedding_vector =
          (float *)g_malloc0(numElements * sizeof(float));
      cudaMemcpy(meta->embedding.embedding_vector,
                 (float *)(tensor_meta->out_buf_ptrs_dev[0]),
                 numElements * sizeof(float), cudaMemcpyDeviceToHost);
      meta->embedding.embedding_length = numElements;
      
      NvDsInferDimsCHW dims;

      /* Access the 1-indexed output layer to get probs */
      getDimsCHWFromDims(dims, tensor_meta->output_layers_info[1].inferDims);
      dual_head_classes = dims.c;

      float *outputCoverageBuffer =
          (float *)tensor_meta->out_buf_ptrs_host[1];
      float maxProbability = 0;
      bool attrFound = false;
      NvDsInferAttribute attr;

      /* Get the output with max probability */
      for (unsigned int c = 0; c < dual_head_classes; c++) {
        float probability = outputCoverageBuffer[c];
        if (probability > 0 && probability > maxProbability) {
          maxProbability = probability;
          attrFound = true;
          attr.attributeIndex = 0;
          attr.attributeValue = c;
          attr.attributeConfidence = probability;
        }
      }
#ifdef GENERATE_DUMMY_META_EXT
      NvDsProductObject *obj =
          (NvDsProductObject *)g_malloc0(sizeof(NvDsProductObject));
      generate_product_meta(obj);

#endif
      /* Generate classifer metadata and attach to obj_meta */
      if (attrFound) {
        NvDsClassifierMeta *classifier_meta =
            nvds_acquire_classifier_meta_from_pool(batch_meta);

        classifier_meta->unique_component_id = tensor_meta->unique_id;

        /* Create NvDsLabel Info*/
        NvDsLabelInfo *label_info =
            nvds_acquire_label_info_meta_from_pool(batch_meta);
        label_info->result_class_id = attr.attributeValue;
        label_info->result_prob = attr.attributeConfidence;

        strcpy(label_info->result_label, labels[label_info->result_class_id]);

        gchar *temp = obj_params->text_params.display_text;
        obj_params->text_params.display_text =
            g_strconcat(temp, " ", label_info->result_label, NULL);
        g_free(temp);

        nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
        nvds_add_classifier_meta_to_object(obj_params, classifier_meta);
        strcpy(dual_head_label, label_info->result_label);
#ifdef GENERATE_DUMMY_META_EXT
        obj->brand = g_strdup(dual_head_label);
        meta->extMsg = obj;
        meta->extMsgSize = sizeof(NvDsProductObject);
#endif

      }

#ifdef GENERATE_DUMMY_META_EXT
      meta->extMsg = obj;
      meta->extMsgSize = sizeof(NvDsProductObject);
#endif

    }
  }
}

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 */
static void
bbox_generated_probe_after_analytics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  NvDsObjectMeta *obj_meta = NULL;
  GstClockTime buffer_pts = 0;
  guint32 stream_id = 0;

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    stream_id = frame_meta->source_id;
    GstClockTime buf_ntp_time = 0;
    if (playback_utc == FALSE) {
      /** Calculate the buffer-NTP-time
       * derived from this stream's RTCP Sender Report here:
       */
      StreamSourceInfo *src_stream = &testAppCtx->streams[stream_id];
      buf_ntp_time = frame_meta->ntp_timestamp;

      if (buf_ntp_time < src_stream->last_ntp_time) {
        NVGSTDS_WARN_MSG_V ("Source %d: NTP timestamps are backward in time."
            " Current: %lu previous: %lu", stream_id, buf_ntp_time,
            src_stream->last_ntp_time);
      }
      src_stream->last_ntp_time = buf_ntp_time;
    }

    float scaleW = 0;
    float scaleH = 0;
    if (!appCtx->config.streammux_config.pipeline_width ||
        !appCtx->config.streammux_config.pipeline_height) {
      g_print("[ERROR] invalid pipeline params\n");

      return;
    }

    scaleW = (float)frame_meta->source_frame_width /
             appCtx->config.streammux_config.pipeline_width;
    scaleH = (float)frame_meta->source_frame_height /
             appCtx->config.streammux_config.pipeline_height;
    
    buffer_pts = frame_meta->buf_pts;
    if (playback_utc == FALSE) {
      /** Use the buffer-NTP-time derived from this stream's RTCP Sender
       * Report here:
       */
      buffer_pts = buf_ntp_time;
    }

    GList *l;
    for (l = frame_meta->obj_meta_list; l != NULL; l = l->next) {
      /* Now using above information we need to form a text that should
       * be displayed on top of the bounding box, so lets form it here. */

      obj_meta = (NvDsObjectMeta *)(l->data);
      NvDsInferTensorMeta *tensor_meta = NULL;
      //! Attaching Embedding tensor metadata
      for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list;
            l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
        if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
          /* convert to tensor metadata */
          tensor_meta = (NvDsInferTensorMeta *)user_meta->user_meta_data;
          //==Retrieve output tensor from model==
          // = (float *)tensor_meta->out_buf_ptrs_host[0];
        }
      }
      /** Generate NvDsEventMsgMeta for every object */
      NvDsEventMsgMeta *msg_meta =
          (NvDsEventMsgMeta *)g_malloc0(sizeof(NvDsEventMsgMeta));

      generate_event_msg_meta(appCtx, msg_meta, obj_meta->class_id, TRUE,
                              buffer_pts,
                              appCtx->config.multi_source_config[stream_id].uri,
                              stream_id, stream_id, obj_meta, scaleW, scaleH,
                              frame_meta, tensor_meta, batch_meta);
      /* Acquire NvDsUserMeta user meta from pool */
      NvDsUserMeta *new_user_meta =
          nvds_acquire_user_meta_from_pool(batch_meta);
      /* Set NvDsUserMeta below */
      new_user_meta->user_meta_data = (void *)msg_meta;
      new_user_meta->base_meta.batch_meta = batch_meta;
      new_user_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
      new_user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)meta_copy_func;
      new_user_meta->base_meta.release_func =
          (NvDsMetaReleaseFunc)meta_free_func;
      /* We want to add NvDsUserMeta to frame level */
      nvds_add_user_meta_to_frame(frame_meta, new_user_meta);
    }
    testAppCtx->streams[stream_id].frameCount++;
  }
}

/** @{ imported from deepstream-app as is */

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

  cintr = TRUE;
}

/**
 * callback function to print the performance numbers of each stream.
 */
static void
perf_cb (gpointer context, NvDsAppPerfStruct * str)
{
  static guint header_print_cnt = 0;
  guint i;
  AppCtx *appCtx = (AppCtx *) context;
  guint numf = str->num_instances;

  g_mutex_lock (&fps_lock);
  guint active_src_count = 0;
  for (i = 0; i < numf; i++) {
    fps[i] = str->fps[i];
    if (fps[i]){
      active_src_count++;
    }
    fps_avg[i] = str->fps_avg[i];
  }
  g_print("Active sources : %u\n", active_src_count);
  if (header_print_cnt % 20 == 0) {
    g_print ("\n**PERF:  ");
    for (i = 0; i < numf; i++) {
      g_print ("FPS %d (Avg)\t", i);
    }
    g_print ("\n");
    header_print_cnt = 0;
  }
  header_print_cnt++;

  time_t t = time (NULL);
  struct tm *tm = localtime (&t);
  printf ("%s", asctime (tm));
  if (num_instances > 1)
    g_print ("PERF(%d): ", appCtx->index);
  else
    g_print ("**PERF:  ");

  for (i = 0; i < numf; i++) {
    g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
  }
  g_print ("\n");
  g_mutex_unlock (&fps_lock);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (quit) {
    return FALSE;
  }

  if (cintr) {
    cintr = FALSE;

    quit = TRUE;
    g_main_loop_quit (main_loop);

    return FALSE;
  }
  return TRUE;
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

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
static void
changemode (int dir)
{
  static struct termios oldt, newt;

  if (dir == 1) {
    tcgetattr (STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON);
    tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  } else
    tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
}

static void
print_runtime_commands (void)
{
  g_print ("\nRuntime commands:\n"
      "\th: Print this help\n"
      "\tq: Quit\n\n" "\tp: Pause\n" "\tr: Resume\n\n");

  if (appCtx[0]->config.tiled_display_config.enable) {
    g_print
        ("NOTE: To expand a source in the 2D tiled display and view object details,"
        " left-click on the source.\n"
        "      To go back to the tiled display, right-click anywhere on the window.\n\n");
  }
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
static gboolean
event_thread_func (gpointer arg)
{
  guint i;
  gboolean ret = TRUE;

  // Check if all instances have quit
  for (i = 0; i < num_instances; i++) {
    if (!appCtx[i]->quit)
      break;
  }

  if (i == num_instances) {
    quit = TRUE;
    g_main_loop_quit (main_loop);
    return FALSE;
  }
  // Check for keyboard input
  if (!kbhit ()) {
    //continue;
    return TRUE;
  }
  int c = fgetc (stdin);
  g_print ("\n");

  gint source_id;
  GstElement *tiler = appCtx[rcfg]->pipeline.tiled_display_bin.tiler;

  if (appCtx[rcfg]->config.tiled_display_config.enable)
  {
    g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

    if (selecting) {
      if (rrowsel == FALSE) {
        if (c >= '0' && c <= '9') {
          rrow = c - '0';
          g_print ("--selecting source  row %d--\n", rrow);
          rrowsel = TRUE;
        }
      } else {
        if (c >= '0' && c <= '9') {
          int tile_num_columns = appCtx[rcfg]->config.tiled_display_config.columns;
          rcol = c - '0';
          selecting = FALSE;
          rrowsel = FALSE;
          source_id = tile_num_columns * rrow + rcol;
          g_print ("--selecting source  col %d sou=%d--\n", rcol, source_id);
          if (source_id >= (gint) appCtx[rcfg]->config.num_source_sub_bins) {
            source_id = -1;
          } else {
            appCtx[rcfg]->show_bbox_text = TRUE;
            appCtx[rcfg]->active_source_index = source_id;
            g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
          }
        }
      }
    }
  }
  switch (c) {
    case 'h':
      print_runtime_commands ();
      break;
    case 'p':
      for (i = 0; i < num_instances; i++)
        pause_pipeline (appCtx[i]);
      break;
    case 'r':
      for (i = 0; i < num_instances; i++)
        resume_pipeline (appCtx[i]);
      break;
    case 'q':
      quit = TRUE;
      g_main_loop_quit (main_loop);
      ret = FALSE;
      break;
    case 'c':
      if (appCtx[rcfg]->config.tiled_display_config.enable && selecting == FALSE && source_id == -1) {
        g_print("--selecting config file --\n");
        c = fgetc(stdin);
        if (c >= '0' && c <= '9') {
          rcfg = c - '0';
          if (rcfg < num_instances) {
            g_print("--selecting config  %d--\n", rcfg);
          } else {
            g_print("--selected config file %d out of bound, reenter\n", rcfg);
            rcfg = 0;
          }
        }
      }
      break;
    case 'z':
      if (appCtx[rcfg]->config.tiled_display_config.enable && source_id == -1 && selecting == FALSE) {
        g_print ("--selecting source --\n");
        selecting = TRUE;
      } else {
        if (!show_bbox_text) {
          GstElement *nvosd = appCtx[rcfg]->pipeline.instance_bins[0].osd_bin.nvosd;
          g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
          g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
        }
        appCtx[rcfg]->active_source_index = -1;
        selecting = FALSE;
        rcfg = 0;
        g_print("--tiled mode --\n");
      }
      break;
    default:
      break;
  }
  return ret;
}

static int
get_source_id_from_coordinates (float x_rel, float y_rel, AppCtx *appCtx)
{
  int tile_num_rows = appCtx->config.tiled_display_config.rows;
  int tile_num_columns = appCtx->config.tiled_display_config.columns;

  int source_id = (int) (x_rel * tile_num_columns);
  source_id += ((int) (y_rel * tile_num_rows)) * tile_num_columns;

  /* Don't allow clicks on empty tiles. */
  if (source_id >= (gint) appCtx->config.num_source_sub_bins)
    source_id = -1;

  return source_id;
}

/**
 * Thread to monitor X window events.
 */
static gpointer
nvds_x_event_thread (gpointer data)
{
  g_mutex_lock (&disp_lock);
  while (display) {
    XEvent e;
    guint index;
    while (XPending (display)) {
      XNextEvent (display, &e);
      switch (e.type) {
        case ButtonPress:
        {
          XWindowAttributes win_attr;
          XButtonEvent ev = e.xbutton;
          gint source_id;
          GstElement *tiler;

          XGetWindowAttributes (display, ev.window, &win_attr);

          for (index = 0; index < MAX_INSTANCES; index++)
            if (ev.window == windows[index])
              break;

          tiler = appCtx[index]->pipeline.tiled_display_bin.tiler;
          g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

          if (ev.button == Button1 && source_id == -1) {
            source_id =
                get_source_id_from_coordinates (ev.x * 1.0 / win_attr.width,
                ev.y * 1.0 / win_attr.height, appCtx[index]);
            if (source_id > -1) {
              g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
              appCtx[index]->active_source_index = source_id;
              appCtx[index]->show_bbox_text = TRUE;
              GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
              g_object_set (G_OBJECT (nvosd), "display-text", TRUE, NULL);
            }
          } else if (ev.button == Button3) {
            g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
            appCtx[index]->active_source_index = -1;
            if (!show_bbox_text) {
              appCtx[index]->show_bbox_text = FALSE;
              GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
              g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
            }
          }
        }
          break;
        case KeyRelease:
        {
          KeySym p, r, q;
          guint i;
          p = XKeysymToKeycode (display, XK_P);
          r = XKeysymToKeycode (display, XK_R);
          q = XKeysymToKeycode (display, XK_Q);
          if (e.xkey.keycode == p) {
            for (i = 0; i < num_instances; i++)
              pause_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == r) {
            for (i = 0; i < num_instances; i++)
              resume_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == q) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
        case ClientMessage:
        {
          Atom wm_delete;
          for (index = 0; index < MAX_INSTANCES; index++)
            if (e.xclient.window == windows[index])
              break;

          wm_delete = XInternAtom (display, "WM_DELETE_WINDOW", 1);
          if (wm_delete != None && wm_delete == (Atom) e.xclient.data.l[0]) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
      }
    }
    g_mutex_unlock (&disp_lock);
    g_usleep (G_USEC_PER_SEC / 20);
    g_mutex_lock (&disp_lock);
  }
  g_mutex_unlock (&disp_lock);
  return NULL;
}

/**
 * callback function to add application specific metadata.
 * Here it demonstrates how to display the URI of source in addition to
 * the text generated after inference.
 */
static gboolean
overlay_graphics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  return TRUE;
}

/**
 * Callback function to notify the status of the model update
 */
static void
infer_model_updated_cb (GstElement * gie, gint err, const gchar * config_file)
{
  double otaTime = 0;
  gettimeofday (&ota_completion_time, NULL);

  otaTime = (ota_completion_time.tv_sec - ota_request_time.tv_sec) * 1000.0;
  otaTime += (ota_completion_time.tv_usec - ota_request_time.tv_usec) / 1000.0;

  const char *err_str = (err == 0 ? "ok" : "failed");
  g_print
      ("\nModel Update Status: Updated model : %s, OTATime = %f ms, result: %s \n\n",
      config_file, otaTime, err_str);
}

/**
 * Function to print detected Inotify handler events
 * Used only for debugging purposes
 */
static void
display_inotify_event (struct inotify_event *i_event)
{
  printf ("    watch decriptor =%2d; ", i_event->wd);
  if (i_event->cookie > 0)
    printf ("cookie =%4d; ", i_event->cookie);

  printf ("mask = ");
  if (i_event->mask & IN_ACCESS)
    printf ("IN_ACCESS ");
  if (i_event->mask & IN_ATTRIB)
    printf ("IN_ATTRIB ");
  if (i_event->mask & IN_CLOSE_NOWRITE)
    printf ("IN_CLOSE_NOWRITE ");
  if (i_event->mask & IN_CLOSE_WRITE)
    printf ("IN_CLOSE_WRITE ");
  if (i_event->mask & IN_CREATE)
    printf ("IN_CREATE ");
  if (i_event->mask & IN_DELETE)
    printf ("IN_DELETE ");
  if (i_event->mask & IN_DELETE_SELF)
    printf ("IN_DELETE_SELF ");
  if (i_event->mask & IN_IGNORED)
    printf ("IN_IGNORED ");
  if (i_event->mask & IN_ISDIR)
    printf ("IN_ISDIR ");
  if (i_event->mask & IN_MODIFY)
    printf ("IN_MODIFY ");
  if (i_event->mask & IN_MOVE_SELF)
    printf ("IN_MOVE_SELF ");
  if (i_event->mask & IN_MOVED_FROM)
    printf ("IN_MOVED_FROM ");
  if (i_event->mask & IN_MOVED_TO)
    printf ("IN_MOVED_TO ");
  if (i_event->mask & IN_OPEN)
    printf ("IN_OPEN ");
  if (i_event->mask & IN_Q_OVERFLOW)
    printf ("IN_Q_OVERFLOW ");
  if (i_event->mask & IN_UNMOUNT)
    printf ("IN_UNMOUNT ");

  if (i_event->mask & IN_CLOSE)
    printf ("IN_CLOSE ");
  if (i_event->mask & IN_MOVE)
    printf ("IN_MOVE ");
  if (i_event->mask & IN_UNMOUNT)
    printf ("IN_UNMOUNT ");
  if (i_event->mask & IN_IGNORED)
    printf ("IN_IGNORED ");
  if (i_event->mask & IN_Q_OVERFLOW)
    printf ("IN_Q_OVERFLOW ");
  printf ("\n");

  if (i_event->len > 0)
    printf ("        name = %s mask= %x \n", i_event->name, i_event->mask);
}

/**
 * Perform model-update OTA operation
 */
void
apply_ota (AppCtx * ota_appCtx)
{
  GstElement *primary_gie = NULL;

  if (ota_appCtx->override_config.primary_gie_config.enable) {
    primary_gie =
        ota_appCtx->pipeline.common_elements.primary_gie_bin.primary_gie;
    gchar *model_engine_file_path =
        ota_appCtx->override_config.primary_gie_config.model_engine_file_path;

    gettimeofday (&ota_request_time, NULL);
    if (model_engine_file_path) {
      g_print ("\nNew Model Update Request %s ----> %s\n",
          GST_ELEMENT_NAME (primary_gie), model_engine_file_path);
      g_object_set (G_OBJECT (primary_gie), "model-engine-file",
          model_engine_file_path, NULL);
    } else {
      g_print
          ("\nInvalid New Model Update Request received. Property model-engine-path is not set\n");
    }
  }
}

/**
 * Independent thread to perform model-update OTA process based on the inotify events
 * It handles currently two scenarios
 * 1) Local Model Update Request (e.g. Standalone Appliation)
 *    In this case, notifier handler watches for the ota_override_file changes
 * 2) Cloud Model Update Request (e.g. EGX with Kubernetes)
 *    In this case, notifier handler watches for the ota_override_file changes along with
 *    ..data directory which gets mounted by EGX deployment in Kubernetes environment.
 */
gpointer
ota_handler_thread (gpointer data)
{

  int length, i = 0;
  char buffer[INOTIFY_EVENT_BUF_LEN];
  OTAInfo *ota = (OTAInfo *) data;
  gchar *ota_ds_config_file = ota->override_cfg_file;
  AppCtx *ota_appCtx = ota->appCtx;
  struct stat file_stat = { 0 };
  GstElement *primary_gie = NULL;
  gboolean connect_pgie_signal = FALSE;

  ota_appCtx->ota_inotify_fd = inotify_init ();

  if (ota_appCtx->ota_inotify_fd < 0) {
    perror ("inotify_init");
    return NULL;
  }

  char *real_path_ds_config_file = realpath (ota_ds_config_file, NULL);
  g_print ("REAL PATH = %s\n", real_path_ds_config_file);

  gchar *ota_dir = g_path_get_dirname (real_path_ds_config_file);
  ota_appCtx->ota_watch_desc =
      inotify_add_watch (ota_appCtx->ota_inotify_fd, ota_dir, IN_ALL_EVENTS);

  int ret = lstat (ota_ds_config_file, &file_stat);
  ret = ret;

  if (S_ISLNK (file_stat.st_mode)) {
    printf (" Override File Provided is Soft Link\n");
    gchar *parent_ota_dir = g_strdup_printf ("%s/..", ota_dir);
    ota_appCtx->ota_watch_desc =
        inotify_add_watch (ota_appCtx->ota_inotify_fd, parent_ota_dir,
        IN_ALL_EVENTS);
  }

  while (1) {
    i = 0;
    length = read (ota_appCtx->ota_inotify_fd, buffer, INOTIFY_EVENT_BUF_LEN);

    if (length < 0) {
      perror ("read");
    }

    if (quit == TRUE)
      goto done;

    while (i < length) {
      struct inotify_event *event = (struct inotify_event *) &buffer[i];

      // Enable below function to print the inotify events, used for debugging purpose
      if (0) {
        display_inotify_event (event);
      }

      if (connect_pgie_signal == FALSE) {
        primary_gie =
            ota_appCtx->pipeline.common_elements.primary_gie_bin.primary_gie;
        if (primary_gie) {
          g_signal_connect (G_OBJECT (primary_gie), "model-updated",
              G_CALLBACK (infer_model_updated_cb), NULL);
          connect_pgie_signal = TRUE;
        } else {
          printf
              ("Gstreamer pipeline element nvinfer is yet to be created or invalid\n");
          continue;
        }
      }

      if (event->len) {
        if (event->mask & IN_MOVED_TO) {
          if (strstr ("..data", event->name)) {
            memset (&ota_appCtx->override_config, 0,
                sizeof (ota_appCtx->override_config));
            if (!IS_YAML(ota_ds_config_file)) {
              if (!parse_config_file (&ota_appCtx->override_config,
                      ota_ds_config_file)) {
                NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                    ota_ds_config_file);
                g_print
                    ("Error: ota_handler_thread: Failed to parse config file '%s'",
                    ota_ds_config_file);
              } else {
                apply_ota (ota_appCtx);
              }
            } else if (IS_YAML(ota_ds_config_file)) {
                if (!parse_config_file_yaml (&ota_appCtx->override_config,
                      ota_ds_config_file)) {
                NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                    ota_ds_config_file);
                g_print
                    ("Error: ota_handler_thread: Failed to parse config file '%s'",
                    ota_ds_config_file);
              } else {
                apply_ota (ota_appCtx);
              }
            }
          }
        }
        if (event->mask & IN_CLOSE_WRITE) {
          if (!(event->mask & IN_ISDIR)) {
            if (strstr (ota_ds_config_file, event->name)) {
              g_print ("File %s modified.\n", event->name);

              memset (&ota_appCtx->override_config, 0,
                  sizeof (ota_appCtx->override_config));
              if (!IS_YAML(ota_ds_config_file)) {
                if (!parse_config_file (&ota_appCtx->override_config,
                        ota_ds_config_file)) {
                  NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                      ota_ds_config_file);
                  g_print
                      ("Error: ota_handler_thread: Failed to parse config file '%s'",
                      ota_ds_config_file);
                } else {
                  apply_ota (ota_appCtx);
                }
              } else if (IS_YAML(ota_ds_config_file)) {
                  if (!parse_config_file_yaml (&ota_appCtx->override_config,
                        ota_ds_config_file)) {
                  NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                      ota_ds_config_file);
                  g_print
                      ("Error: ota_handler_thread: Failed to parse config file '%s'",
                      ota_ds_config_file);
                } else {
                  apply_ota (ota_appCtx);
                }
              }
            }
          }
        }
      }
      i += INOTIFY_EVENT_SIZE + event->len;
    }
  }
done:
  inotify_rm_watch (ota_appCtx->ota_inotify_fd, ota_appCtx->ota_watch_desc);
  close (ota_appCtx->ota_inotify_fd);

  free (real_path_ds_config_file);
  g_free (ota_dir);

  g_free (ota);
  return NULL;
}

/**
 * Parse labels from label file and fill them in the global variable: `labels`
 */
void get_labels_from_file(gchar *classifier_label_file) {
  // Use file pointer
  FILE *file = fopen(classifier_label_file, "r");

  // Return if file is not found
  if (file == NULL) {
    perror("Error opening file.");
    return;
  }

  guint line_count = 0;
  while (fgets(labels[line_count], MAX_CHAR_LENGTH_PER_LINE, file) != NULL) {
    // Remove newline character from the end of the line
    labels[line_count][strcspn(labels[line_count], "\n")] = '\0';
    line_count++;

    if (line_count == dual_head_classes)
      break;
    else if (line_count > dual_head_classes) {
      g_print(
          "[WARNING] You have more labels than classes. Labels after line %d "
          "will not be "
          "used.\n",
          dual_head_classes);
      break;
    }
  }

  if (line_count < dual_head_classes)
    g_print(
        "[WARNING] You have more classes than labels. Empty strings will be "
        "used in "
        "metadata field when a class without label is predicted.\n");

  // Close the file
  fclose(file);
}

/** @} imported from deepstream-app as is */

int
main (int argc, char *argv[])
{
  testAppCtx = (TestAppCtx *) g_malloc0 (sizeof (TestAppCtx));
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;
  guint i;
  OTAInfo *otaInfo = NULL;

  ctx = g_option_context_new ("Nvidia DeepStream MDX Perception App");
  group = g_option_group_new ("abc", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());

  GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    g_print ("%s",g_option_context_get_help (ctx, TRUE, NULL));
    return -1;
  }

  if (print_version) {
    g_print ("deepstream-MDXPerception-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    return 0;
  }

  if (print_dependencies_version) {
    g_print ("deepstream-MDXPerception-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    return 0;
  }

  if (cfg_files) {
    num_instances = g_strv_length (cfg_files);
  }
  if (input_files) {
    num_input_files = g_strv_length (input_files);
  }

  if (!cfg_files || num_instances == 0) {
    NVGSTDS_ERR_MSG_V ("Specify config file with -c option");
    return_value = -1;
    goto done;
  }

  for (i = 0; i < num_instances; i++) {
    appCtx[i] = (AppCtx *) g_malloc0 (sizeof (AppCtx));
    appCtx[i]->person_class_id = -1;
    appCtx[i]->car_class_id = -1;
    appCtx[i]->index = i;
    appCtx[i]->active_source_index = -1;
    if (show_bbox_text) {
      appCtx[i]->show_bbox_text = TRUE;
    }

    if (input_files && input_files[i]) {
      appCtx[i]->config.multi_source_config[0].uri =
          g_strdup_printf ("file://%s", input_files[i]);
      g_free (input_files[i]);
    }

    if(IS_YAML(cfg_files[i])) {
      if (!parse_config_file_yaml (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
    } else {
      if (!parse_config_file (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
    }

    if (override_cfg_file && override_cfg_file[i]) {
      if (!g_file_test (override_cfg_file[i],
            (GFileTest)(G_FILE_TEST_IS_REGULAR | G_FILE_TEST_IS_SYMLINK)))
      {
        g_print ("Override file %s does not exist, quitting...\n",
            override_cfg_file[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
      otaInfo = (OTAInfo *) g_malloc0 (sizeof (OTAInfo));
      otaInfo->appCtx = appCtx[i];
      otaInfo->override_cfg_file = override_cfg_file[i];
      appCtx[i]->ota_handler_thread = g_thread_new ("ota-handler-thread",
          ota_handler_thread, otaInfo);
    }
  }

  for (i = 0; i < num_instances; i++) {
    for (guint j = 0; j < appCtx[i]->config.num_source_sub_bins; j++) {
       /** Force the source (applicable only if RTSP)
        * to use TCP for RTP/RTCP channels.
        * forcing TCP to avoid problems with UDP port usage from within docker-
        * container.
        * The UDP RTCP channel when run within docker had issues receiving
        * RTCP Sender Reports from server
        */
      if (force_tcp)
        appCtx[i]->config.multi_source_config[j].select_rtp_protocol = 0x04;
    }
    if (!create_pipeline (appCtx[i], bbox_generated_probe_after_analytics,
            NULL, perf_cb, overlay_graphics)) {
      NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
      return_value = -1;
      goto done;
    }
    /** Now add probe to RTPSession plugin src pad */
    for (guint j = 0; j < appCtx[i]->pipeline.multi_src_bin.num_bins; j++) {
      testAppCtx->streams[j].id = j;
    }
    /** In test5 app, as we could have several sources connected
     * for a typical IoT use-case, raising the nvstreammux's
     * buffer-pool-size to 16 */
    g_object_set (appCtx[i]->pipeline.multi_src_bin.streammux,
        "buffer-pool-size", STREAMMUX_BUFFER_POOL_SIZE, NULL);
  }

  if (model_used == APP_CONFIG_ANALYTICS_RETAIL_DUAL_HEAD_SGIE) {
    NvDsGieConfig *classifier_config =
        &appCtx[0]->config.secondary_gie_sub_bin_config[0];
    gboolean classifier_enabled = classifier_config->enable;

    if (!classifier_enabled) {
      perror("Dual Head Classifier is disabled.");
      return 1;
    }

    gchar *classifier_label_file = classifier_config->label_file_path;
    get_labels_from_file(classifier_label_file);
  }

  main_loop = g_main_loop_new (NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, NULL);

  g_mutex_init (&disp_lock);
  display = XOpenDisplay (NULL);
  for (i = 0; i < num_instances; i++) {
    guint j;

    if (!show_bbox_text) {
      GstElement *nvosd = appCtx[i]->pipeline.instance_bins[0].osd_bin.nvosd;
      g_object_set(G_OBJECT(nvosd), "display-text", FALSE, NULL);
    }

    if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
            GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
      NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
      return_value = -1;
      goto done;
    }

    for (j = 0; j < appCtx[i]->config.num_sink_sub_bins; j++) {
      XTextProperty xproperty;
      gchar *title;
      guint width, height;
      XSizeHints hints = {0};

      if (!GST_IS_VIDEO_OVERLAY (appCtx[i]->pipeline.instance_bins[0].sink_bin.
              sub_bins[j].sink)) {
        continue;
      }

      if (!display) {
        NVGSTDS_ERR_MSG_V ("Could not open X Display");
        return_value = -1;
        goto done;
      }

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width)
        width =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width;
      else
        width = appCtx[i]->config.tiled_display_config.width;

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height)
        height =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height;
      else
        height = appCtx[i]->config.tiled_display_config.height;

      width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
      height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;

      hints.flags = PPosition | PSize;
      hints.x = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_x;
      hints.y = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_y;
      hints.width = width;
      hints.height = height;

      windows[i] =
          XCreateSimpleWindow (display, RootWindow (display,
              DefaultScreen (display)), hints.x, hints.y, width, height, 2,
              0x00000000, 0x00000000);

      XSetNormalHints(display, windows[i], &hints);

      if (num_instances > 1)
        title = g_strdup_printf (APP_TITLE "-%d", i);
      else
        title = g_strdup (APP_TITLE);
      if (XStringListToTextProperty ((char **) &title, 1, &xproperty) != 0) {
        XSetWMName (display, windows[i], &xproperty);
        XFree (xproperty.value);
      }

      XSetWindowAttributes attr = { 0 };
      if ((appCtx[i]->config.tiled_display_config.enable &&
              appCtx[i]->config.tiled_display_config.rows *
              appCtx[i]->config.tiled_display_config.columns == 1) ||
          (appCtx[i]->config.tiled_display_config.enable == 0)) {
        attr.event_mask = KeyRelease;
      } else if (appCtx[i]->config.tiled_display_config.enable) {
        attr.event_mask = ButtonPress | KeyRelease;
      }
      XChangeWindowAttributes (display, windows[i], CWEventMask, &attr);

      Atom wmDeleteMessage = XInternAtom (display, "WM_DELETE_WINDOW", False);
      if (wmDeleteMessage != None) {
        XSetWMProtocols (display, windows[i], &wmDeleteMessage, 1);
      }
      XMapRaised (display, windows[i]);
      XSync (display, 1);       //discard the events for now
      gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (appCtx
              [i]->pipeline.instance_bins[0].sink_bin.sub_bins[j].sink),
          (gulong) windows[i]);
      gst_video_overlay_expose (GST_VIDEO_OVERLAY (appCtx[i]->pipeline.
              instance_bins[0].sink_bin.sub_bins[j].sink));
      if (!x_event_thread)
        x_event_thread = g_thread_new ("nvds-window-event-thread",
            nvds_x_event_thread, NULL);
    }
  }

  /* Dont try to set playing state if error is observed */
  if (return_value != -1) {
    for (i = 0; i < num_instances; i++) {
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {

        g_print ("\ncan't set pipeline to playing state.\n");
        return_value = -1;
        goto done;
      }
    }
  }

  print_runtime_commands ();

  changemode (1);

  g_timeout_add (40, event_thread_func, NULL);
  g_main_loop_run (main_loop);

  changemode (0);

done:

  g_print ("Quitting\n");
  for (i = 0; i < num_instances; i++) {
    if (appCtx[i] == NULL)
      continue;

    if (appCtx[i]->return_value == -1)
      return_value = -1;

    destroy_pipeline (appCtx[i]);

    if (appCtx[i]->ota_handler_thread && override_cfg_file[i]) {
      inotify_rm_watch (appCtx[i]->ota_inotify_fd, appCtx[i]->ota_watch_desc);
      g_thread_join (appCtx[i]->ota_handler_thread);
    }

    g_mutex_lock (&disp_lock);
    if (windows[i])
      XDestroyWindow (display, windows[i]);
    windows[i] = 0;
    g_mutex_unlock (&disp_lock);

    g_free (appCtx[i]);
  }

  g_mutex_lock (&disp_lock);
  if (display)
    XCloseDisplay (display);
  display = NULL;
  g_mutex_unlock (&disp_lock);
  g_mutex_clear (&disp_lock);

  if (main_loop) {
    g_main_loop_unref (main_loop);
  }

  if (ctx) {
    g_option_context_free (ctx);
  }

  if (return_value == 0) {
    g_print ("App run successful\n");
  } else {
    g_print ("App run failed\n");
  }

  gst_deinit ();

  return return_value;

  g_free (testAppCtx);

  return 0;
}

static gchar *
get_first_result_label (NvDsClassifierMeta * classifierMeta)
{
  GList *n;
  for (n = classifierMeta->label_info_list; n != NULL; n = n->next) {
    NvDsLabelInfo *labelInfo = (NvDsLabelInfo *) (n->data);
    if (labelInfo->result_label[0] != '\0') {
      return g_strdup (labelInfo->result_label);
    }
  }
  return NULL;
}

static void
schema_fill_sample_sgie_vehicle_metadata (NvDsObjectMeta * obj_params,
    NvDsVehicleObject * obj)
{
  if (!obj_params || !obj) {
    return;
  }

  /** The JSON obj->classification, say type, color, or make
   * according to the schema shall have null (unknown)
   * classifications (if the corresponding sgie failed to provide a label)
   */
  obj->type = NULL;
  obj->make = NULL;
  obj->model = NULL;
  obj->color = NULL;
  obj->license = NULL;
  obj->region = NULL;

  GList *l;
  for (l = obj_params->classifier_meta_list; l != NULL; l = l->next) {
    NvDsClassifierMeta *classifierMeta = (NvDsClassifierMeta *) (l->data);
    switch (classifierMeta->unique_component_id) {
      case SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID:
        obj->type = get_first_result_label (classifierMeta);
        break;
      case SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID:
        obj->color = get_first_result_label (classifierMeta);
        break;
      case SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID:
        obj->make = get_first_result_label (classifierMeta);
        break;
      default:
        break;
    }
  }
}
