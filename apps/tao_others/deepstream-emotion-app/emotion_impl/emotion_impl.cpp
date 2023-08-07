/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <sys/stat.h>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <string>
#include <condition_variable>
#include <sstream>
#include <map>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "gst-nvevent.h"

#include "nvdscustomlib_base.hpp"

#include "cv/emotions/Emotions.h"
#include "ds_facialmark_meta.h"
#include "nvdsinfer_context.h"
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

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

std::map<size_t, string> emotionsList;

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

void emotion_impl_logger(NvDsInferContextHandle handle, unsigned int unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {

    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
        GST_ERROR("Emotion impl [UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        GST_WARNING("Emotion impl[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_INFO:
        GST_INFO("Emotion impl[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_DEBUG:
        GST_DEBUG("Emotion impl[UID %d]: %s", unique_id, log_message);
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

class EmotionAlgorithm : public DSCustomLibraryBase
{
public:
  EmotionAlgorithm() {
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
  ~EmotionAlgorithm();

private:
  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

public:
  guint source_id = 0;
  guint m_frameNum = 0;
  bool outputthread_stopped = false;

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

  void *m_scratchNvBufSurface = NULL;

  // Currently dumps first 5 input video frame into file for demonstration
  // purpose. Use vooya or simillar player to view NV12/RGBA video raw frame
  int dump_max_frames = 5;

  std::unique_ptr<cvcore::emotions::Emotions> m_emotionsPtr;
  std::string m_config_file_path;
};

// Create Custom Algorithm / Library Context
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params)
{
  return new EmotionAlgorithm();
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
         cvcore::ModelInferenceParams eMotionInferenceParams,
         cvcore::ModelInputParams ModelInputParams,
         struct EtltModelParams EngineGenParam,
         std::string etlt_path)
{

  strncpy(InferCtxParams->tltEncodedModelFilePath, etlt_path.c_str(), etlt_path.size());
  strncpy(InferCtxParams->tltModelKey, EngineGenParam.decodeKey.c_str(), EngineGenParam.decodeKey.size());
  //The model is NHWC
  InferCtxParams->inferInputDims.c = ModelInputParams.inputLayerHeight;
  InferCtxParams->inferInputDims.h = ModelInputParams.inputLayerWidth;
  InferCtxParams->inferInputDims.w = get_channel_from_imagetype(ModelInputParams.modelInputType);
  InferCtxParams->numOutputLayers = eMotionInferenceParams.outputLayers.size();
  InferCtxParams->outputLayerNames = new char*[eMotionInferenceParams.outputLayers.size()];

  for(int i=0; i < eMotionInferenceParams.outputLayers.size(); i++) {
    InferCtxParams->outputLayerNames[i] = new char[eMotionInferenceParams.outputLayers[i].size()+1]();
    strncpy(InferCtxParams->outputLayerNames[i], eMotionInferenceParams.outputLayers[i].c_str(), eMotionInferenceParams.outputLayers[i].size());
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
      nullptr, emotion_impl_logger);
  if(status != NVDSINFER_SUCCESS) {
    GST_ERROR("generate trt engine failed \n");
    return false;
  }
  return true;
}

// Set Init Parameters
bool EmotionAlgorithm::SetInitParams(DSCustom_CreateParams *params)
{
  DSCustomLibraryBase::SetInitParams(params);

  GstStructure *s1 = NULL;

  s1 = gst_caps_get_structure(m_inCaps, 0);


  cvcore::ModelInferenceParams eMotionInferenceParams =
  {
    "emotions_fp16_b32.engine",          /**< Path to the engine */
    {"input_landmarks:0"},               /**< Input layer name */
    {"softmax/Softmax:0"},               /**< Output layer name */
  };

  cvcore::ModelInputParams ModelInputParams =
  {
    32,     /**< Max Batch Size */
    136,    /**< Width of the model input */
    1,      /**< Height of the model input */
    cvcore::Y_F32   /**< Format of the model input */
  };

  struct EtltModelParams EngineGenParam = {
    {"nvidia_tlt"},
    NvDsInferNetworkMode_FP16
  };

  NvDsInferContextInitParams *InferCtxParams = new NvDsInferContextInitParams();
  memset(InferCtxParams, 0, sizeof (*InferCtxParams));
  InferCtxParams->autoIncMem = 1;
  InferCtxParams->maxGPUMemPer = 90;

  ifstream fconfig;
  std::map<string, float> model_params_list;
  std::string etlt_path;
  /* Parse emotion model config file*/
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
              eMotionInferenceParams.engineFilePath = s;
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
        if (config["label"]) {
            YAML::Node primes = config["label"];
            for (YAML::const_iterator it=primes.begin();it!=primes.end();++it) {

              std::string seq = it->as<std::string>();
              std::vector<std::string> label_strs = split(seq.c_str(), ",");
              if ( !(label_strs[0].empty())  && ! (label_strs[1].empty()) ) {
                  std::istringstream index_str(label_strs[0]);
                  size_t index;
                  index_str >> index;
                  emotionsList[index]=label_strs[1];
              }
            }
        }
      } else {
        fconfig.open(m_config_file_path);
        if (!fconfig.is_open()) {
            g_print("The model config file %s open is failed!\n", m_config_file_path.c_str());
            return -1;
        }

        while (!fconfig.eof()) {
            string strParam;
	        if (getline(fconfig, strParam)) {
            std::vector<std::string> param_strs = split(strParam, "=");
            float value;
            if (param_strs.size() < 2)
              continue;
            if(!(param_strs[0].empty()) && !(param_strs[1].empty())) {
              if(param_strs[0] == "enginePath"){
                  string s = get_absolute_path(param_strs[1]);
                  struct stat buffer;
                  if(stat (s.c_str(), &buffer) == 0){
                    eMotionInferenceParams.engineFilePath = s;
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
              } else if (param_strs[0] == "label") {
                  std::vector<std::string> label_strs = split(param_strs[1], ",");
                  if ( !(label_strs[0].empty())  && ! (label_strs[1].empty()) ) {
                      std::istringstream index_str(label_strs[0]);
                      size_t index;
                      index_str >> index;
                      emotionsList[index]=label_strs[1];
                  }
              } else {
                  std::istringstream isStr(param_strs[1]);
                  isStr >> value;
                  model_params_list[param_strs[0]] = value;
              }
            }
          }
	    }
      }
  }
  fconfig.close();
  
  if (model_params_list.count("inputHeight")) {
    ModelInputParams.inputLayerHeight = model_params_list["inputHeight"];
  }

  if (model_params_list.count("inputWidth")) {
    ModelInputParams.inputLayerWidth = model_params_list["inputWidth"];
  }

  if (model_params_list.count("inputType")) {
    if (model_params_list["inputType"] < cvcore::NV12) {
        int input_type = model_params_list["inputType"];
        ModelInputParams.modelInputType = (cvcore::ImageType)input_type;
    }        
  }

  if (model_params_list.count("batchSize")) {
    ModelInputParams.maxBatchSize = model_params_list["batchSize"];
  }

  string engine_path =  eMotionInferenceParams.engineFilePath;
 
  if (access(engine_path.c_str(), F_OK ) == -1) {
      get_infer_params(InferCtxParams, eMotionInferenceParams,
          ModelInputParams, EngineGenParam, etlt_path);
      std::string devId = std::string("gpu0");
      engine_path = etlt_path +  "_b" + std::to_string(ModelInputParams.maxBatchSize) + "_" +
         devId + "_" + networkMode2Str(EngineGenParam.networkMode) + ".engine";
      if (access(engine_path.c_str(), F_OK ) == -1) {
          if (!generate_trt_engine(InferCtxParams)) {
              GST_ERROR("build engine failed \n");
              return false;
          }

	      if (access( engine_path.c_str(), F_OK ) == -1) {
              // Still no named engine found, check the degradingn engines
              if(EngineGenParam.networkMode == NvDsInferNetworkMode_INT8) {
                  engine_path = etlt_path +  "_b" + std::to_string(ModelInputParams.maxBatchSize) + "_" +
                            devId + "_" + networkMode2Str(NvDsInferNetworkMode_FP16) + ".engine";
                  if (access( engine_path.c_str(), F_OK ) == -1) {
                      //Degrade again
                      engine_path = etlt_path +  "_b" + std::to_string(ModelInputParams.maxBatchSize) + "_" +
                          devId + "_" + networkMode2Str(NvDsInferNetworkMode_FP32) + ".engine";
                      if (access( engine_path.c_str(), F_OK ) == -1) {
                          //failed
                          GST_ERROR("No proper engine generated %s\n", engine_path.c_str());
                          return false;
                      }
                  }
              } else if (EngineGenParam.networkMode == NvDsInferNetworkMode_FP16) {
                  engine_path = etlt_path +  "_b" + std::to_string(ModelInputParams.maxBatchSize) + "_" +
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

  eMotionInferenceParams.engineFilePath = engine_path;

  std::unique_ptr<cvcore::emotions::Emotions> emotionsInstance(
    new cvcore::emotions::Emotions(
    cvcore::emotions::defaultPreProcessorParams,
    ModelInputParams,
    eMotionInferenceParams));

  m_emotionsPtr = std::move(emotionsInstance);

  m_outputThread = new std::thread(&EmotionAlgorithm::OutputThread, this);

  return true;
}

// Return Compatible Output Caps based on input caps
GstCaps* EmotionAlgorithm::GetCompatibleCaps (GstPadDirection direction,
        GstCaps* in_caps, GstCaps* othercaps)
{
  GstCaps* result = NULL;
  GstStructure *s1, *s2;
  gint width, height;
  gint i, num, denom;
  const gchar *inputFmt = NULL;

  GST_DEBUG ("\n----------\ndirection = %d (1=Src, 2=Sink) -> %s:\nCAPS ="
  " %s\n", direction, __func__, gst_caps_to_string(in_caps));
  GST_DEBUG ("%s : OTHERCAPS = %s\n", __func__, gst_caps_to_string(othercaps));

  othercaps = gst_caps_truncate(othercaps);
  othercaps = gst_caps_make_writable(othercaps);

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
      gst_structure_get_int (s1, "width", &width);
      gst_structure_get_int (s1, "height", &height);

      /* otherwise the dimension of the output heatmap needs to be fixated */

      // Here change the width and height on output caps based on the
      // information provided byt the custom library
      gst_structure_fixate_field_nearest_int(s2, "width", width);
      gst_structure_fixate_field_nearest_int(s2, "height", height);
      if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
      {
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num,
          denom);
      }

      gst_structure_set (s2, "width", G_TYPE_INT, (gint)(width), NULL);
      gst_structure_set (s2, "height", G_TYPE_INT, (gint)(height) , NULL);
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

char *EmotionAlgorithm::QueryProperties ()
{
    char *str = new char[1000];
    strcpy (str, "EMOTION LIBRARY PROPERTIES\n \t\t\tcustomlib-props=\"config-file\" : path of the model config file");
    return str;
}

bool EmotionAlgorithm::HandleEvent (GstEvent *event)
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
            //g_print ("waiting for processq to be empty, buffers in processq
            // = %ld\n", m_processQ.size());
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

// Set Custom Library Specific Properties
bool EmotionAlgorithm::SetProperty(Property &prop)
{
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
EmotionAlgorithm::~EmotionAlgorithm()
{
  std::unique_lock<std::mutex> lk(m_processLock);
  m_processCV.wait(lk, [&]{return m_processQ.empty();});
  m_stop = TRUE;
  m_processCV.notify_all();
  lk.unlock();

  /* Wait for OutputThread to complete */
  if (m_outputThread) {
    m_outputThread->join();
  }

  if (m_scratchNvBufSurface)
  {
    cudaFree (&m_scratchNvBufSurface);
    m_scratchNvBufSurface = NULL;
  }
}

/* Process Buffer */
BufferResult EmotionAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GstMapInfo in_map_info;

  GST_DEBUG("CustomLib: ---> Inside %s frame_num = %d\n", __func__,
  m_frameNum++);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;

  // Add custom preprocessing logic if required, here
  // Pass the buffer to output_loop for further processing and pusing to next component
  // Currently its just dumping few decoded video frames

  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}

/* Output Processing Thread */
void EmotionAlgorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  NvBufSurface *outSurf = NULL;
  NvDsBatchMeta *batch_meta = NULL;
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
    // Once buffer processing is done, push the buffer to the downstream by using gst_pad_push function
    batch_meta = gst_buffer_get_nvds_batch_meta (packetInfo.inbuf);

    //First getting the bbox of faces and eyes
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;
    uint8_t *imagedataPtr = NULL;

    nvds_acquire_meta_lock (batch_meta);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
        obj_meta = (NvDsObjectMeta *) (l_obj->data);
        if (!obj_meta)
          continue;

        for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list;
          l_user != NULL; l_user = l_user->next) {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
          if(user_meta->base_meta.meta_type ==
            (NvDsMetaType)NVDS_USER_RIVA_META_FACEMARK) {

            NvDsFacePointsMetaData *facepoints_meta =
              (NvDsFacePointsMetaData *)user_meta->user_meta_data;
            cvcore::Array<cvcore::ArrayN<cvcore::Vector2f,
              cvcore::emotions::Emotions::NUM_FACIAL_LANDMARKS>> landmarks(1);
            landmarks.setSize(1);
            cvcore::Array<cvcore::ArrayN<float,
              cvcore::emotions::Emotions::TOP_EMOTIONS>> results(1, true);
            results.setSize(1);
            cvcore::Array<cvcore::ArrayN<std::size_t,cvcore::emotions::Emotions::TOP_EMOTIONS>> topEmotions(1,true);
            topEmotions.setSize(1);
            for(std::size_t j = 0; j < cvcore::emotions::Emotions::
              NUM_FACIAL_LANDMARKS; ++j){
              landmarks[0][j].x = facepoints_meta->mark[j].x;
              landmarks[0][j].y = facepoints_meta->mark[j].y;
            }
            m_emotionsPtr->execute(results, topEmotions, landmarks);
            /* The emotions model is a classifier, the classifier meta
             * can store the output labels */
            float confidence = 0.0;
            for (int count=0; count < results[0].getSize(); count++){
              if(results[0][count] > confidence)
                confidence = results[0][count];
            }
            g_stpcpy(obj_meta->obj_label, emotionsList[topEmotions[0][0]].c_str());
            NvDsClassifierMeta *classifier_meta =
              nvds_acquire_classifier_meta_from_pool (batch_meta);
            NvDsLabelInfo *label_info =
              nvds_acquire_label_info_meta_from_pool (batch_meta);
            label_info->label_id = 1;
            label_info->result_class_id = 1;
            label_info->result_prob = confidence;
            g_strlcpy (label_info->result_label, emotionsList
              [topEmotions[0][0]].c_str(),
              MAX_LABEL_SIZE);
            gchar *temp = obj_meta->text_params.display_text;
            obj_meta->text_params.display_text =
              g_strconcat (temp, " ", emotionsList[topEmotions[0][0]].c_str(), nullptr);
            g_free (temp);
            nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);
          }
        }
      }
    }
    nvds_release_meta_lock (batch_meta);

    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);

    // Transform IP case
    outSurf = in_surf;
    outBuffer = packetInfo.inbuf;

    // Output buffer parameters checking
    if (outSurf->numFilled != 0)
    {
      g_assert ((guint)m_outVideoInfo.width == outSurf->surfaceList->width);
      g_assert ((guint)m_outVideoInfo.height == outSurf->surfaceList->height);
    }

    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element),
      outBuffer);
    GST_DEBUG ("CustomLib: %s in_surf=%p, Pushing Frame %d to downstream..."
      " flow_ret = %d TS=%" GST_TIME_FORMAT " \n", __func__, in_surf,
      packetInfo.frame_num, flow_ret,
      GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

    lk.lock();
    continue;
  }
  outputthread_stopped = true;
  lk.unlock();
  return;
}
