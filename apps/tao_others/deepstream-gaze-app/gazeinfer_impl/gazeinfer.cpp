/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"

#include "nvdscustomlib_base.hpp"
#include "nvdsinfer_context.h"

#include "cuda_runtime_api.h"
#include "cudaEGL.h"

#include "ds_facialmark_meta.h"
#include "ds_gaze_meta.h"
#include "cv/gazenet/GazeNet.h"
#include "cv/core/Memory.h"
#include "ds_yml_parse.h"
#include <yaml-cpp/yaml.h>

using namespace std;
using std::string;

#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"

inline bool CHECK_(int e, int iLine, const char *szFile) {
    if (e != cudaSuccess) {
      std::cout << "CUDA runtime error " << e << " at line " << iLine <<
        " in file " << szFile;
        exit (-1);
        return false;
    }
    return true;
}
#define ck(call) CHECK_(call, __LINE__, __FILE__)

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
const cvcore::CameraIntrinsics cameraParams = {960.5037478246579, 960.1507947013808, 643.5179717135431,
                                               365.1361007229173};

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

static std::string get_absolute_path(std::string path) {
    if (path == "" || path[0] == '/') {
      /*Empty or Abs path, return as is. */
      return path;
    }

    /* Rel path. Get lib path and append rel_path to it */
    Dl_info dl_info;
    dladdr(reinterpret_cast<void *>(get_absolute_path), &dl_info);
    std::string lib_path = dl_info.dli_fname;
    std::size_t pos = lib_path.find_last_of('/');
    std::string lib_dir_path = lib_path.substr(0, pos + 1);
    return lib_dir_path + path;
}

void gaze_impl_logger(NvDsInferContextHandle handle, unsigned int unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {

    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
        GST_ERROR("GazeNet impl [UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        GST_WARNING("GazeNet impl[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_INFO:
        GST_INFO("GazeNet impl[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_DEBUG:
        GST_DEBUG("GazeNet impl[UID %d]: %s", unique_id, log_message);
        return;
  }
}

struct EtltModelParams {
  string decodeKey;
  NvDsInferNetworkMode networkMode;
};

static std::string
networkMode2Str(const NvDsInferNetworkMode type)
{
    switch (type)
    {
        case NvDsInferNetworkMode_FP32:
            return "fp32";
        case NvDsInferNetworkMode_INT8:
            return "int8";
        case NvDsInferNetworkMode_FP16:
            return "fp16";
        default:
            return "UNKNOWN";
    }
}

class GazeAlgorithm : public DSCustomLibraryBase
{
public:
  GazeAlgorithm() {
    m_vectorProperty.clear();
    outputthread_stopped = false;
  }

  /* Set Init Parameters */
  virtual bool SetInitParams(DSCustom_CreateParams *params);

  /* Set Custom Properties  of the library */
  virtual bool SetProperty(Property &prop);

  /* Pass GST events to the library */
  virtual bool HandleEvent(GstEvent *event);

  virtual char *QueryProperties ();

  /* Process Incoming Buffer */
  virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

  /* Retrun Compatible Caps */
  virtual GstCaps * GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps);

  /* Deinit members */
  ~GazeAlgorithm();

private:

  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

public:
  guint source_id = 0;
  guint m_frameNum = 0;
  bool outputthread_stopped = false;

  /* Custom Library Bufferpool */
  //GstBufferPool *m_dsBufferPool = NULL;

  /* Output Thread Pointer */
  std::thread *m_outputThread = NULL;

  /* Queue and Lock Management */
  std::queue<PacketInfo> m_processQ;
  std::mutex m_processLock;
  std::condition_variable m_processCV;

  /* Aysnc Stop Handling */
  gboolean m_stop = FALSE;

  /* Vector Containing Key:Value Pair of Custom Lib Properties */
  std::vector<Property> m_vectorProperty;

  int m_batch_width = 0;
  int m_batch_height = 0;
  int m_gpu_id = 0;
  NvBufSurface m_temp_surf;
  NvBufSurface *m_process_surf;
  cudaStream_t m_convertStream;
  NvBufSurfTransformParams m_transform_params;
  NvBufSurfTransformConfigParams m_transform_config_params;
  std::unique_ptr<cvcore::gazenet::GazeNet> m_objGaze;
  std::unique_ptr<cvcore::gazenet::GazeNetVisualizer> m_objGazeVisualizer;
  std::string m_config_file_path;
  CUgraphicsResource m_cuda_resource;
  CUeglFrame m_egl_frame;
};

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params);
// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params)
{
  return new GazeAlgorithm();
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
            s.erase(s.find_last_not_of('\r') + 1);
            s.erase(s.find_last_not_of(" ") + 1);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

unsigned int get_channel_from_imagetype(cvcore::ImageType type)
{
    unsigned int channel = 0;
    switch (type) {
      case cvcore::Y_U8:
      case cvcore::Y_U16:
      case cvcore::Y_F16:
      case cvcore::Y_F32:
        channel = 1;
        break;
      case cvcore::RGB_U8:
      case cvcore::RGB_U16:
      case cvcore::RGB_F16:
      case cvcore::RGB_F32:
      case cvcore::BGR_U8:
      case cvcore::BGR_U16:
      case cvcore::BGR_F16:
      case cvcore::BGR_F32:
      case cvcore::PLANAR_RGB_U8:
      case cvcore::PLANAR_RGB_U16:
      case cvcore::PLANAR_RGB_F16:
      case cvcore::PLANAR_RGB_F32:
      case cvcore::PLANAR_BGR_U8:
      case cvcore::PLANAR_BGR_U16:
      case cvcore::PLANAR_BGR_F16:
      case cvcore::PLANAR_BGR_F32:
      case cvcore::NV12:
        channel = 3;
        break;
      case cvcore::RGBA_U8:
      case cvcore::RGBA_U16:
      case cvcore::RGBA_F16:
      case cvcore::RGBA_F32:
      case cvcore::PLANAR_RGBA_U8:
      case cvcore::PLANAR_RGBA_U16:
      case cvcore::PLANAR_RGBA_F16:
      case cvcore::PLANAR_RGBA_F32:
         channel = 4;
         break;
      default:
         channel = 0;
         break;
    }
    return channel;
}

NvDsInferFormat get_format_from_imagetype(cvcore::ImageType type)
{
    NvDsInferFormat format;
    switch (type)
    {
      case cvcore::RGB_U8:
      case cvcore::RGB_U16:
      case cvcore::RGB_F16:
      case cvcore::RGB_F32:
      case cvcore::PLANAR_RGB_U8:
      case cvcore::PLANAR_RGB_U16:
      case cvcore::PLANAR_RGB_F16:
      case cvcore::PLANAR_RGB_F32:
        format = NvDsInferFormat_RGB;
        break;
      case cvcore::BGR_U8:
      case cvcore::BGR_U16:
      case cvcore::BGR_F16:
      case cvcore::BGR_F32:
      case cvcore::PLANAR_BGR_U8:
      case cvcore::PLANAR_BGR_U16:
      case cvcore::PLANAR_BGR_F16:
      case cvcore::PLANAR_BGR_F32:
        format = NvDsInferFormat_BGR;
        break;
      case cvcore::Y_U8:
      case cvcore::Y_U16:
      case cvcore::Y_F16:
      case cvcore::Y_F32:
        format = NvDsInferFormat_GRAY;
        break;
      case cvcore::RGBA_U8:
      case cvcore::RGBA_U16:
      case cvcore::RGBA_F16:
      case cvcore::RGBA_F32:
      case cvcore::PLANAR_RGBA_U8:
      case cvcore::PLANAR_RGBA_U16:
      case cvcore::PLANAR_RGBA_F16:
      case cvcore::PLANAR_RGBA_F32:
        format = NvDsInferFormat_RGBA;
        break;
      case cvcore::NV12:
        format = NvDsInferFormat_Tensor;
        break;
      default:
        format = NvDsInferFormat_Unknown;
        break;
    }
    return format;
}

void get_infer_params(NvDsInferContextInitParams *InferCtxParams,
         cvcore::ModelInferenceParams InferenceParams,
         cvcore::ModelInputParams ModelInputParams,
         struct EtltModelParams EngineGenParam,
         std::string etlt_path)
{

  strncpy(InferCtxParams->tltEncodedModelFilePath, etlt_path.c_str(), etlt_path.size());
  strncpy(InferCtxParams->tltModelKey, EngineGenParam.decodeKey.c_str(), EngineGenParam.decodeKey.size());
  //The model is NHWC
  InferCtxParams->inferInputDims.c = get_channel_from_imagetype(ModelInputParams.modelInputType);
  InferCtxParams->inferInputDims.h = ModelInputParams.inputLayerHeight;
  InferCtxParams->inferInputDims.w = ModelInputParams.inputLayerWidth;
  InferCtxParams->numOutputLayers = InferenceParams.outputLayers.size();
  InferCtxParams->outputLayerNames = new char*[InferenceParams.outputLayers.size()];

  for(int i=0; i < InferenceParams.outputLayers.size(); i++) {
    InferCtxParams->outputLayerNames[i] = new char[InferenceParams.outputLayers[i].size()+1]();
    strncpy(InferCtxParams->outputLayerNames[i], InferenceParams.outputLayers[i].c_str(), InferenceParams.outputLayers[i].size());
  }
  InferCtxParams->maxBatchSize = ModelInputParams.maxBatchSize;
  InferCtxParams->networkMode = EngineGenParam.networkMode;
  InferCtxParams->uniqueID = 3;
  InferCtxParams->outputBufferPoolSize = 16;
  InferCtxParams->networkInputFormat = get_format_from_imagetype(ModelInputParams.modelInputType);
}

bool generate_trt_engine(NvDsInferContextInitParams *InferCtxParams)
{
  NvDsInferContextHandle ctx_handle;
  NvDsInferStatus status =
      createNvDsInferContext (&ctx_handle, *InferCtxParams,
      nullptr, gaze_impl_logger);
  if(status != NVDSINFER_SUCCESS) {
    GST_ERROR("generate trt engine failed \n");
    return false;
  }
  return true;
}

// Set Init Parameters
bool GazeAlgorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);
  GstStructure *s1 = NULL;
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  cudaError_t cudaReturn;
  ifstream fconfig;

  s1 = gst_caps_get_structure(m_inCaps, 0);

  m_gpu_id = params->m_gpuId;

  NvBufSurfaceCreateParams create_params = { 0 };

  create_params.gpuId = m_gpu_id;
  create_params.width = m_batch_width;
  create_params.height = m_batch_height;
  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_BGR;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_DEFAULT;

  if (NvBufSurfaceCreate (&m_process_surf, 1,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvinfer");
    return false;
  }


  if(m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (NvBufSurfaceMapEglImage (m_process_surf, 0) != 0) {
      GST_ERROR ("Error:Could not map EglImage from NvBufSurface for nvinfer");
      return false;
    }

    if (cuGraphicsEGLRegisterImage (&m_cuda_resource,
        m_process_surf->surfaceList[0].mappedAddr.eglImage,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
            ("Failed to register EGLImage in cuda\n"), (NULL));
        return false;
    }
    if (cuGraphicsResourceGetMappedEglFrame (&m_egl_frame,
        m_cuda_resource, 0, 0) != CUDA_SUCCESS) {
        GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
            ("Failed to get mapped EGL Frame\n"), (NULL));
        return false;
    }
  }

  m_transform_params.src_rect = new NvBufSurfTransformRect[1];
  m_transform_params.dst_rect = new NvBufSurfTransformRect[1];
  m_transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  m_transform_params.transform_flip = NvBufSurfTransform_None;
  m_transform_params.transform_filter = NvBufSurfTransformInter_Default;

  m_transform_config_params.compute_mode = NvBufSurfTransformCompute_GPU;

  cudaReturn =
    cudaStreamCreateWithFlags (&m_convertStream, cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (m_element, RESOURCE, FAILED,
        ("Failed to create cuda stream"),
        ("cudaStreamCreateWithFlags failed with error %s",
            cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  m_transform_config_params.gpu_id = m_gpu_id;
  m_transform_config_params.cuda_stream = m_convertStream;

  m_temp_surf.surfaceList = new NvBufSurfaceParams[1];
  m_temp_surf.batchSize = 1;
  m_temp_surf.gpuId = m_gpu_id;

  m_outputThread = new std::thread(&GazeAlgorithm::OutputThread, this);

  cvcore::ModelInferenceParams GazeInferenceParams =
  {
    "gazenet_facegrid_fp16_b8.engine",                /**< Path to the engine*/
    {"input_left_images:0", "input_right_images:0",
    "input_face_images:0", "input_facegrid:0"},       /**< Input layer names */
    {"fc_joint/concat:0"},                            /**< Output layer name */
  };

  cvcore::ModelInputParams GazeModelInputParams = {8, 224, 224, cvcore::Y_U8};

  struct EtltModelParams EngineGenParam = {
    {"nvidia_tlt"},
    NvDsInferNetworkMode_FP16
  };
  std::string etlt_path;
  NvDsInferContextInitParams *InferCtxParams = new NvDsInferContextInitParams();
  memset(InferCtxParams, 0, sizeof (*InferCtxParams));

  if (!m_config_file_path.empty()) {
      /* Parse model config file*/
      if ( g_str_has_suffix(m_config_file_path.c_str(), ".yml")
        || (g_str_has_suffix(m_config_file_path.c_str(), ".yaml"))) {

        YAML::Node config = YAML::LoadFile(m_config_file_path.c_str());

        if (config.IsNull()) {
            g_printerr ("config file is NULL.\n");
            return -1;
        }

        if (config["enginePath"]) {
            string s =
              get_absolute_path(config["enginePath"].as<std::string>().c_str());
            struct stat buffer;
            if(stat (s.c_str(), &buffer) == 0){
              GazeInferenceParams.engineFilePath = s;
            }
        }
        if (config["etltPath"]) {
            etlt_path =
              get_absolute_path(config["etltPath"].as<std::string>().c_str());
        }
        if (config["etltKey"]) {
            EngineGenParam.decodeKey = config["etltKey"].as<std::string>().c_str();
        }
        if (config["networkMode"]) {
            std::string type_name = config["networkMode"].as<std::string>();
            if (type_name.c_str() == "fp16")
                EngineGenParam.networkMode = NvDsInferNetworkMode_FP16;
            else if (type_name.c_str() == "fp32")
                EngineGenParam.networkMode = NvDsInferNetworkMode_FP32;
            else if (type_name.c_str() == "int8")
                EngineGenParam.networkMode = NvDsInferNetworkMode_INT8;
        }
      } else {
        fconfig.open(m_config_file_path);
        if (!fconfig.is_open()) {
          g_print("The model config file open is failed!\n");
          return -1;
        }

        while (!fconfig.eof()) {
          string strParam;
	      if (getline(fconfig, strParam)) {
              std::vector<std::string> param_strs = split(strParam, "=");
              if (param_strs.size() < 2)
                  continue;
              if(!(param_strs[0].empty()) && !(param_strs[1].empty())) {
                  if (param_strs[0] == "enginePath"){
                      string s = get_absolute_path(param_strs[1]);
                      struct stat buffer;
                      if(stat (s.c_str(), &buffer) == 0){
                          GazeInferenceParams.engineFilePath = s;
                      }
                  } else if(param_strs[0] == "etltPath") {
                      etlt_path = get_absolute_path(param_strs[1]);
                  } else if(param_strs[0] == "etltKey") {
                      EngineGenParam.decodeKey = param_strs[1];
                  } else if(param_strs[0] == "networkMode") {
                      if (param_strs[1] == "fp16")
                          EngineGenParam.networkMode = NvDsInferNetworkMode_FP16;
                      else if (param_strs[1] == "fp32")
                          EngineGenParam.networkMode = NvDsInferNetworkMode_FP32;
                      else if (param_strs[1] == "int8")
                          EngineGenParam.networkMode = NvDsInferNetworkMode_INT8;
                  }
              }
	      }
        }
        fconfig.close();
      }

      string engine_path =  GazeInferenceParams.engineFilePath;
 
      if (access( engine_path.c_str(), F_OK ) == -1) {
          if (access( etlt_path.c_str(), F_OK ) == -1) {
              GST_ERROR("No model available in %s\n", etlt_path.c_str());
              return false;
          }
          get_infer_params(InferCtxParams, GazeInferenceParams,
              GazeModelInputParams, EngineGenParam, etlt_path);
          std::string devId = std::string("gpu0");
          engine_path = etlt_path +  "_b" + std::to_string(cvcore::gazenet::defaultModelInputParams.maxBatchSize) + "_" +
              devId + "_" + networkMode2Str(EngineGenParam.networkMode) + ".engine";
          if (access( engine_path.c_str(), F_OK ) == -1) {
              if(!generate_trt_engine(InferCtxParams)) {
                  GST_ERROR("build engine failed \n");
                  return false;
              }
	      if (access( engine_path.c_str(), F_OK ) == -1) {
                  // Still no named engine found, check the degradingn engines
                  if(EngineGenParam.networkMode == NvDsInferNetworkMode_INT8) {
                      engine_path = etlt_path +  "_b" + std::to_string(cvcore::gazenet::defaultModelInputParams.maxBatchSize) + "_" +
                           devId + "_" + networkMode2Str(NvDsInferNetworkMode_FP16) + ".engine";
                      if (access( engine_path.c_str(), F_OK ) == -1) {
                          //Degrade again
                          engine_path = etlt_path +  "_b" + std::to_string(cvcore::gazenet::defaultModelInputParams.maxBatchSize) + "_" +
                           devId + "_" + networkMode2Str(NvDsInferNetworkMode_FP32) + ".engine";
                          if (access( engine_path.c_str(), F_OK ) == -1) {
                              //failed
                              GST_ERROR("No proper engine generated %s\n", engine_path.c_str());
                              return false;
                          }
                      }
                  } else if (EngineGenParam.networkMode == NvDsInferNetworkMode_FP16) {
                      engine_path = etlt_path +  "_b" + std::to_string(cvcore::gazenet::defaultModelInputParams.maxBatchSize) + "_" +
                           devId + "_" + networkMode2Str(NvDsInferNetworkMode_FP32) + ".engine";
                      if (access( engine_path.c_str(), F_OK ) == -1) {
                          //failed
                          GST_ERROR("No proper engine generated %s\n", engine_path.c_str());
                          return false;
                      }
                  }
              }
          }
      }

      GazeInferenceParams.engineFilePath = engine_path;
    
  }

  std::unique_ptr<cvcore::gazenet::GazeNet> objGaze(new 
     cvcore::gazenet::GazeNet(cvcore::gazenet::defaultPreProcessorParams,
     GazeModelInputParams, GazeInferenceParams));
  std::unique_ptr<cvcore::gazenet::GazeNetVisualizer> objGazeVisualizer(
        new cvcore::gazenet::GazeNetVisualizer(cameraParams));

  m_objGaze = std::move(objGaze);
  m_objGazeVisualizer = std::move(objGazeVisualizer);

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* GazeAlgorithm::GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps)
{
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  //gint width, height;
  gint i, num, denom;
  const gchar *inputFmt = NULL;

  GST_DEBUG ("\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\n"
      "CAPS = %s\n", direction, __func__, gst_caps_to_string(in_caps));
  GST_DEBUG ("%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

  //num_input_caps = gst_caps_get_size (in_caps);
  int num_output_caps = gst_caps_get_size (othercaps);

  // TODO: Currently it only takes first caps
  s1 = gst_caps_get_structure(in_caps, 0);
  for (i=0; i<num_output_caps; i++)
  {
    s2 = gst_caps_get_structure(othercaps, i);
    inputFmt = gst_structure_get_string (s1, "format");

    GST_DEBUG ("InputFMT = %s \n\n", inputFmt);

    // Check for desired color format
    if ((strncmp(inputFmt, FORMAT_NV12, strlen(FORMAT_NV12)) == 0) ||
            (strncmp(inputFmt, FORMAT_RGBA, strlen(FORMAT_RGBA)) == 0))
    {
      //Set these output caps
      gst_structure_get_int (s1, "width", &m_batch_width);
      gst_structure_get_int (s1, "height", &m_batch_height);

      /* otherwise the dimension of the output heatmap needs to be fixated */

      // Here change the width and height on output caps based on the
      // information provided by the custom library
      gst_structure_fixate_field_nearest_int(s2, "width", m_batch_width);
      gst_structure_fixate_field_nearest_int(s2, "height", m_batch_height);
      if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
      {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num,
            denom);
      }

      // TODO: Get width, height, coloutformat, and framerate from
      // customlibrary API set the new properties accordingly
      gst_structure_set (s2, "width", G_TYPE_INT, (gint)(m_batch_width), NULL);
      gst_structure_set (s2, "height", G_TYPE_INT, (gint)(m_batch_height),
          NULL);
      gst_structure_set (s2, "format", G_TYPE_STRING, inputFmt, NULL);

      result = gst_caps_ref(othercaps);
      gst_caps_unref(othercaps);
      GST_DEBUG ("%s : Updated OTHERCAPS = %s \n\n", __func__,
          gst_caps_to_string(othercaps));

      break;
    } else {
      continue;
    }
  }
  return result;
}

bool GazeAlgorithm::HandleEvent (GstEvent *event)
{
  switch (GST_EVENT_TYPE(event))
  {
       case GST_EVENT_EOS:
           m_processLock.lock();
           m_stop = TRUE;
           m_processCV.notify_all();
           m_processLock.unlock();
           while (outputthread_stopped == FALSE)
           {
               g_usleep (1000);
           }
           break;
       default:
           break;
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_STREAM_EOS)
  {
      gst_nvevent_parse_stream_eos (event, &source_id);
  }
  return true;
}

char *GazeAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "GAZENET LIBRARY PROPERTIES\n \t\t\tconfig-file=GAZENET MODEL CONFIG FILE\n");
    return str;
}

// Set Custom Library Specific Properties
bool GazeAlgorithm::SetProperty(Property &prop)
{
  std::cout << "Inside Custom Lib : Setting Prop Key=" << prop.key << " Value="
      << prop.value << std::endl;
  m_vectorProperty.emplace_back(prop.key, prop.value);

  try
  {
    if (prop.key.compare("config-file") == 0) {
      m_config_file_path.assign(prop.value);
    }
  }
  catch(std::invalid_argument& e)
  {
      std::cout << "Invalid engine file path" << std::endl;
      return false;
  }

  return true;
}

/* Deinitialize the Custom Lib context */
GazeAlgorithm::~GazeAlgorithm()
{
  std::unique_lock<std::mutex> lk(m_processLock);
  //std::cout << "Process Q Empty : " << m_processQ.empty() << std::endl;
  m_processCV.wait(lk, [&]{return m_processQ.empty();});
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }

  if(m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
      cuGraphicsUnregisterResource (m_cuda_resource);
  }

  if(m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY)
      NvBufSurfaceUnMapEglImage (m_process_surf, 0);

  if (m_process_surf)
      NvBufSurfaceDestroy (m_process_surf);


  delete[] m_transform_params.src_rect;
  delete[] m_transform_params.dst_rect;
  delete[] m_temp_surf.surfaceList;

  if (m_convertStream)
    cudaStreamDestroy (m_convertStream);
}

/* Process Buffer */
BufferResult GazeAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GST_DEBUG("GazeInfer: ---> Inside %s frame_num = %d\n", __func__,
      m_frameNum++);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;


  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}

/* Output Processing Thread */
void GazeAlgorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  NvBufSurface *outSurf = NULL;
  int num_in_meta = 0;
  NvDsBatchMeta *batch_meta = NULL;
  
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  std::unique_lock<std::mutex> lk(m_processLock);
  /* Run till signalled to stop. */
  while (1) {

    /* Wait if processing queue is empty. */
    if (m_processQ.empty()) {
      if (m_stop == TRUE) {
        break;
      }
      m_processCV.wait(lk);
      continue;
    }

    PacketInfo packetInfo = m_processQ.front();
    m_processQ.pop();

    m_processCV.notify_all();
    lk.unlock();

    // Add custom algorithm logic here
    // Once buffer processing is done, push the buffer to the downstream by
    // using gst_pad_push function

    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);

    batch_meta = gst_buffer_get_nvds_batch_meta (packetInfo.inbuf);
    if (!batch_meta) {
      GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
        ("%s:No batch meta available", __func__), (NULL));
      return;
    }
    num_in_meta = batch_meta->num_frames_in_batch;
  
    //First getting the bbox of faces and eyes
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    uint8_t *imagedataPtr = NULL;

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
       l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
      
      m_transform_params.src_rect[0] =
          {0, 0, in_surf->surfaceList[frame_meta->batch_id].width,
          in_surf->surfaceList[frame_meta->batch_id].height};
      m_transform_params.dst_rect[0] =
          {0, 0, m_batch_width, m_batch_height};

      err = NvBufSurfTransformSetSessionParams (&m_transform_config_params);
      if (err != NvBufSurfTransformError_Success) {
          GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
          ("NvBufSurfTransformSetSessionParams failed with error %d", err),
              (NULL));
          return;
      }

      m_temp_surf.surfaceList[0] = in_surf->surfaceList[frame_meta->batch_id];
      m_temp_surf.numFilled=1;
      m_temp_surf.memType=in_surf->memType;

      /* Convert the frame data into BGR planar format since the      */
      /* cvcore::gazenet::defaultPreProcessorParams set the format BGR*/

      if (m_process_surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
          imagedataPtr = (uint8_t *)m_egl_frame.frame.pPitch[0];
      } else {
          imagedataPtr = (uint8_t *)m_process_surf->surfaceList[0].dataPtr;
      }
      
      err = NvBufSurfTransform (&m_temp_surf, m_process_surf,
          &m_transform_params);
      if (err != NvBufSurfTransformError_Success) {
          GST_ELEMENT_ERROR (m_element, STREAM, FAILED, 
            ("NvBufSurfTransform failed with error %d while converting "
            "buffer\n",err), (NULL));
          return;
      }

      m_temp_surf.numFilled = 0;

      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
         l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);
          if (!obj_meta)
            continue;

          bool landmark_flag = false;
          cvcore::Array<cvcore::ArrayN<cvcore::Vector2f,
              cvcore::gazenet::GazeNetPreProcessor::NUM_LANDMARKS>>
              gazeInputLandmarks(1, true);
          gazeInputLandmarks.setSize(1);
          for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
              l_user != NULL; l_user = l_user->next) {
              NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
              if(user_meta->base_meta.meta_type ==
                (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {
                  NvDsFacePointsMetaData *facepoints_meta =
                    (NvDsFacePointsMetaData *)user_meta->user_meta_data;
                  if (!facepoints_meta)
                      break;
                  if (gazeInputLandmarks[0].getSize() >
                      facepoints_meta->facemark_num)
                      break;
                  landmark_flag = true;
                  for (int i = 0; i < gazeInputLandmarks[0].getSize(); i++)
                  {
                      gazeInputLandmarks[0][i].x = facepoints_meta->mark[i].x;
                      gazeInputLandmarks[0][i].y = facepoints_meta->mark[i].y;
                  }
              }
          }

          //Only face object with landmarks can be inferred
          if (landmark_flag) {
              cvcore::Array<cvcore::BBox> gazeFaceBBoxes(1);
              gazeFaceBBoxes.setSize(1);
              cvcore::Array<cvcore::Vector2i> leftEndPt(1, true);
              cvcore::Array<cvcore::Vector2i> rightEndPt(1, true);
              cvcore::Array<cvcore::Vector2i> leftStartPt(1, true);
              cvcore::Array<cvcore::Vector2i> rightStartPt(1, true);
              leftEndPt.setSize(1);
              rightEndPt.setSize(1);
              leftStartPt.setSize(1);
              rightStartPt.setSize(1);

      
              cvcore::Array<cvcore::ArrayN<float,
                  cvcore::gazenet::GazeNet::OUTPUT_SIZE>> gazeOutput(1, true);
              gazeOutput.setSize(1);
          
              cvcore::Tensor<cvcore::NHWC, cvcore::C3, cvcore::U8>
                  inputImageDevice(
                  (size_t)m_process_surf->surfaceList[0].width,
                  (size_t)m_process_surf->surfaceList[0].height, (size_t)1,
                  (size_t)m_process_surf->surfaceList[0].pitch, imagedataPtr,
                  false);

              gazeFaceBBoxes[0].xmin = obj_meta->rect_params.left;
              gazeFaceBBoxes[0].xmax = obj_meta->rect_params.left +
                  obj_meta->rect_params.width -1;
              gazeFaceBBoxes[0].ymin = obj_meta->rect_params.top;
              gazeFaceBBoxes[0].ymax = obj_meta->rect_params.top +
                  obj_meta->rect_params.height -1;

              m_objGaze->execute(gazeOutput, inputImageDevice, gazeFaceBBoxes,
                  gazeInputLandmarks);
              m_objGazeVisualizer->execute(leftEndPt, leftStartPt, rightEndPt, rightStartPt, gazeInputLandmarks,
                                       gazeOutput);
              nvds_add_gaze_meta (batch_meta, obj_meta, gazeOutput[0], leftStartPt, leftEndPt,
                  rightStartPt, rightEndPt);
          }
      }
    }

    // Transform IP case
    outSurf = in_surf;
    outBuffer = packetInfo.inbuf;

    // Output buffer parameters checking
    if (outSurf->numFilled != 0)
    {
        g_assert ((guint)m_outVideoInfo.width == outSurf->surfaceList->width);
        g_assert ((guint)m_outVideoInfo.height == outSurf->surfaceList->height);
    }

    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element), outBuffer);
    GST_DEBUG ("CustomLib: %s in_surf=%p, Pushing Frame %d to downstream..."
        " flow_ret = %d TS=%" GST_TIME_FORMAT " \n",  __func__, in_surf,
        packetInfo.frame_num, flow_ret, GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

    lk.lock();
    continue;
  }
  outputthread_stopped = true;
  lk.unlock();
  return;
}
