/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "postprocesslib_impl.h"
#include <filesystem>

using namespace std;

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

/* This quark is required to identify NvDsMeta when iterating through
 * the buffer metadatas */
static GQuark _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);

extern "C" IDSPostProcessLibrary *CreateCustomAlgoCtx(DSPostProcess_CreateParams *params)
{
  return new PostProcessAlgorithm(params);
}


/*Separate a config file entry with delimiters
 *to be able to parse it.*/
std::vector<std::string>
PostProcessAlgorithm::SplitString (std::string input) {
 std::stringstream longStr(input);
  std::string item;
  std::vector <std::string> ret;
  while (std::getline (longStr, item, ';')){
    ret.push_back(item);
  }
  return ret;
}

std::set<gint>
PostProcessAlgorithm::SplitStringInt (std::string input) {

  std::stringstream longStr(input);
  std::string item;
  std::set <gint> ret;
  while (std::getline (longStr, item, ';')){
    ret.insert(stoi(item));
  }
  return ret;
}

/* Get the absolute path of a file mentioned in the config given a
 * file path absolute/relative to the config file. */

bool
PostProcessAlgorithm::GetAbsFilePath (
    const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar abs_real_file_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  /* Absolute path. No need to resolve further. */
  if (file_path[0] == '/') {
    /* Check if the file exists, return error if not. */
    if (!realpath (file_path, abs_real_file_path)) {
      /* Ignore error if file does not exist and use the unresolved path. */
      if (errno != ENOENT)
        return FALSE;
    }
    g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
    return TRUE;
  }

  /* Get the absolute path of the config file. */
  if (!realpath (cfg_file_path, abs_cfg_path)) {
    return FALSE;
  }

  /* Remove the file name from the absolute path to get the directory of the
   * config file. */
  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  /* Get the absolute file path from the config file's directory path and
   * relative file path. */
  abs_file_path = g_strconcat (abs_cfg_path, file_path, nullptr);

  /* Resolve the path.*/
  if (realpath (abs_file_path, abs_real_file_path) == nullptr) {
    /* Ignore error if file does not exist and use the unresolved path. */
    if (errno == ENOENT)
      g_strlcpy (abs_real_file_path, abs_file_path, _PATH_MAX);
    else {
      g_free (abs_file_path);
      return FALSE;
    }
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
bool PostProcessAlgorithm::ParseLabelsFile(
    const std::string& labelsFilePath) {
    std::ifstream labels_file(labelsFilePath, std::ios_base::in);
    std::string delim{';'};
    if (!labels_file) {
        printError("Could not open labels file:%s", safeStr(labelsFilePath));
        return false;
    }
    while (labels_file.good() && !labels_file.eof()) {
        std::string line, word;
        std::vector<std::string> l;
        size_t pos = 0, oldpos = 0;

        std::getline(labels_file, line, '\n');
        if (line.empty())
            continue;

        while ((pos = line.find(delim, oldpos)) != std::string::npos) {
            word = line.substr(oldpos, pos - oldpos);
            l.push_back(word);
            oldpos = pos + delim.length();
        }
        l.push_back(line.substr(oldpos));
        m_Labels.push_back(l);
    }

    if (labels_file.bad()) {
        printError("Failed to parse labels file:%s, iostate:%d",
            safeStr(labelsFilePath), (int)labels_file.rdstate());
        return false;
    }
    return true;
}


bool PostProcessAlgorithm::SetConfigFile (const gchar *cfg_file_path){
  bool ret = true;
  NvDsPostProcessStatus status = NVDSPOSTPROCESS_SUCCESS;

  if (!cfg_file_path || !std::filesystem::exists(cfg_file_path))
  {
    printError("Config File input not provided or doesn't exist");
    return false;
  }
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);
  if (!(configyml.size() > 0)) {
    printError("Unable to parse config file '%s' ", cfg_file_path);
    return false;
  }

  m_processLock.lock();
  // Parse the config file here
  if(configyml["property"]) {
   for(YAML::const_iterator itr = configyml["property"].begin();
        itr != configyml["property"].end(); ++itr){
      std::string paramKey = itr->first.as<std::string>();
      if (paramKey == "gpu-id"){
        m_gpuId = itr->second.as<gint>();
        m_initParams.gpuID = m_gpuId;
      }
      else if (paramKey == "preprocessor-support"){
        m_preprocessor_support =  itr->second.as<gboolean>();
        m_initParams.preprocessor_support = m_preprocessor_support;
      }
      else if (paramKey == "network-type"){
        switch (itr->second.as<gint>()) {
          case NvDsPostProcessNetworkType_Detector:
          case NvDsPostProcessNetworkType_Classifier:
          case NvDsPostProcessNetworkType_Segmentation:
          case NvDsPostProcessNetworkType_InstanceSegmentation:
          case NvDsPostProcessNetworkType_BodyPose:
          case NvDsPostProcessNetworkType_Other:
            m_networkType = static_cast<NvDsPostProcessNetworkType>(itr->second.as<gint>());
            break;
          default:
            g_printerr ("Error. Invalid value for 'network-type':'%d'\n",
                itr->second.as<gint>());
            return false;
            break;
        }
        m_initParams.networkType = m_networkType;
      }
      else if (paramKey == "process-mode"){
        m_processMode = itr->second.as<gint>();
      }
      else if (paramKey == "num-detected-classes"){
        m_numDetectedClasses = itr->second.as<gint>();
        m_initParams.numDetectedClasses = m_numDetectedClasses;
      }
      else if (paramKey == "gie-unique-id"){
        m_gieUniqueId = itr->second.as<gint>();
        m_initParams.uniqueID = m_gieUniqueId;
      }
      else if (paramKey == "labelfile-path"){
        ret = ParseLabelsFile(itr->second.as<std::string>());
        if (ret ==false){
          return ret;
        }
        std::strncpy(m_initParams.labelsFilePath,
            (itr->second.as<std::string>()).c_str(), sizeof(m_initParams.labelsFilePath)-1);
      }
      else if (paramKey == "cluster-mode"){
        switch (itr->second.as<gint>())
        {
          case 0:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES;
            break;
          case 1:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_DBSCAN;
            break;
          case 2:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_NMS;
            break;
          case 3:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_DBSCAN_NMS_HYBRID;
            break;
          case 4:
            m_clusterMode = NVDSPOSTPROCESS_CLUSTER_NONE;
            break;
          default:
            g_printerr ("Error. Invalid value for 'cluster-mode':'%d'\n",
                itr->second.as<gint>());
            return false;
            break;
        }
        m_initParams.clusterMode = m_clusterMode;
      }
      else if (paramKey == "release-tensor-meta"){
        m_releaseTensorMeta = itr->second.as<gboolean>();
      }
      else if (paramKey == "output-instance-mask"){
        m_outputInstanceMask = itr->second.as<gboolean>();
      }
      else if (paramKey == "is-classifier"){
        m_isClassifier = itr->second.as<gboolean>();
      }
      else if (paramKey == "classifier-threshold"){
        m_classifierThreshold = itr->second.as<gfloat>();
        m_initParams.classifierThreshold = m_classifierThreshold;
      }
      else if (paramKey == "classifier-type"){
        m_classifierType = itr->second.as<std::string>();
        g_free (m_initParams.classifier_type);
        m_initParams.classifier_type = g_strdup(m_classifierType.c_str());
      }
      else if (paramKey == "segmentation-threshold"){
        m_segmentationThreshold = itr->second.as<gfloat>();
        m_initParams.segmentationThreshold = m_segmentationThreshold;
      }
      else if (paramKey == "segmentation-output-order"){
        m_initParams.segmentationOutputOrder =
          static_cast<NvDsPostProcessTensorOrder>(itr->second.as<gint>());
      }
      else if (paramKey == "parse-classifier-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customClassifierParseFuncName, temp.c_str(),
            sizeof(m_initParams.customClassifierParseFuncName)-1);
      }
      else if (paramKey == "parse-bbox-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customBBoxParseFuncName, temp.c_str(),
            sizeof(m_initParams.customBBoxParseFuncName)-1);
      } else if (paramKey == "parse-bbox-instance-mask-func-name") {
        std::string temp = itr->second.as<std::string>();
        std::strncpy (m_initParams.customBBoxInstanceMaskParseFuncName, temp.c_str(),
            sizeof(m_initParams.customBBoxInstanceMaskParseFuncName)-1);
      }
      else if (paramKey == "output-blob-names"){
        m_outputBlobNames = SplitString (itr->second.as<std::string>());
        gchar **values;
        int len = (int) m_outputBlobNames.size();
        if (m_initParams.outputLayerNames){
          for (guint i=0; i < m_initParams.numOutputLayers; i++){
            g_free(m_initParams.outputLayerNames[i]);
            m_initParams.outputLayerNames[i] = NULL;
          }
          g_free (m_initParams.outputLayerNames);
        }
        values = g_new (gchar *, len + 1);
        for (int i = 0; i < len; i++) {
          int size = 64;
          char* str2 = (char*) g_malloc0(sizeof(char) * size);
          std::strncpy (str2, m_outputBlobNames[i].c_str(), size-1);
          values[i] = str2;
        }
        values[len] = NULL;
        m_initParams.outputLayerNames = values;
        m_initParams.numOutputLayers = len;
      }
      else if (paramKey == "operate-on-class-ids"){
        m_operateOnClassIds = SplitStringInt (itr->second.as<std::string>());
      }
      else if (paramKey == "filter-out-class-ids"){
        m_filterOutClassIds = SplitStringInt (itr->second.as<std::string>());
      }
      else {
        printWarning ("Unknown parameter %s ",paramKey.c_str());
      }
   }
  }
  else {
    printError("property group not present in config file '%s' ", cfg_file_path);
    m_processLock.unlock();
    return false;
  }

  if (m_initParams.networkType == NvDsPostProcessNetworkType_Detector ||
      m_initParams.networkType == NvDsPostProcessNetworkType_InstanceSegmentation){
    NvDsPostProcessDetectionParams detection_params{DEFAULT_PRE_CLUSTER_THRESHOLD,
        DEFAULT_POST_CLUSTER_THRESHOLD, DEFAULT_EPS,
        DEFAULT_GROUP_THRESHOLD, DEFAULT_MIN_BOXES,
        DEFAULT_DBSCAN_MIN_SCORE, DEFAULT_NMS_IOU_THRESHOLD, DEFAULT_TOP_K,
        0, 0, 0, 0, 0, 0,
        {TRUE, (NvOSD_ColorParams){1.0,0.0,0.0,1.0},
         FALSE,(NvOSD_ColorParams){1.0,0.0,0.0,1.0}}};
    detection_params.color_params.have_border_color = TRUE;
    detection_params.color_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};
    detection_params.color_params.have_bg_color = FALSE;

    /* Parse the parameters for "all" classes if the group has been specified.
     * Detection/Segmentation */
    if (configyml["class-attrs-all"]) {
      ret = ParseConfAttr (configyml["class-attrs-all"], -1, detection_params);
      if (ret ==false){
        printError("Parsing 'class-attrs-all' group failed");
        return ret;
      }
    }

    /* Initialize the per-class vector with the same default/parsed values for
     * all classes. */
    if (m_initParams.perClassDetectionParams){
      delete [] m_initParams.perClassDetectionParams;
    }
    m_initParams.perClassDetectionParams =
      new NvDsPostProcessDetectionParams[m_initParams.numDetectedClasses];

    for (uint32_t icnt = 0; icnt < m_initParams.numDetectedClasses; icnt++){
      m_initParams.perClassDetectionParams[icnt] = detection_params;
    }

    for(YAML::const_iterator itr = configyml.begin(); itr != configyml.end(); ++itr) {
      std::string paramKey = itr->first.as<std::string>();
      std::string class_str = "class-attrs-";
      if ((paramKey != "class-attrs-all") &&
          (paramKey.size() >= class_str.size()))  {
        if (class_str.compare(0,class_str.size(),paramKey.c_str(),
              class_str.size()) == 0) {
          std::string num_str = paramKey.substr(class_str.size());
          gint64 class_index = stoi(num_str);
          m_initParams.perClassDetectionParams[class_index] = detection_params;
          ret = ParseConfAttr (configyml[paramKey], class_index,
              m_initParams.perClassDetectionParams[class_index]);
          if (ret ==false){
            printError("Parsing '%s' group failed",paramKey.c_str());
            return ret;
          }
        }
      }
    }
    status = preparePostProcess();
    if (status != NVDSPOSTPROCESS_SUCCESS){
      return false;
    }
  }
  else if (m_initParams.networkType == NvDsPostProcessNetworkType_Classifier ||
           m_initParams.networkType == NvDsPostProcessNetworkType_Segmentation ||
           m_initParams.networkType ==  NvDsPostProcessNetworkType_BodyPose){

    status = preparePostProcess();
    if (status != NVDSPOSTPROCESS_SUCCESS){
      return false;
    }
  }
  else {
    printError("Parsing for network type %d is not supported",
        m_initParams.networkType);
    return false;
  }

  m_processLock.unlock();
  return true;
}

NvDsPostProcessStatus
PostProcessAlgorithm::preparePostProcess(){
  NvDsPostProcessStatus ret = NVDSPOSTPROCESS_CONFIG_FAILED;

  m_Postprocessor.reset();
  switch (m_initParams.networkType){
  case NvDsPostProcessNetworkType_BodyPose:
    m_Postprocessor = std::make_unique<BodyPoseModelPostProcessor>(m_gieUniqueId, m_gpuId);
    ret = m_Postprocessor->initResource(m_initParams);
    break;
    //FIXME:
  case NvDsPostProcessNetworkType_Other:
    printWarning(" Failed to validate the network type not supported, %d",m_initParams.networkType);
    return ret;
  default:
    printError(" Failed to validate the network type, unknown network %d",m_initParams.networkType);
    return ret;
  }
  if (ret != NVDSPOSTPROCESS_SUCCESS){
    m_Postprocessor.reset();
  }
  return ret;
}

bool PostProcessAlgorithm::ParseConfAttr (YAML::Node node, gint64 class_index,
    NvDsPostProcessDetectionParams& params)
{
  bool ret = true;

  for(YAML::const_iterator itr = node.begin(); itr != node.end(); ++itr) {

    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "detected-min-w"){
      params.detectionMinWidth = itr->second.as<gint>();
    }
    else if (paramKey == "detected-min-h"){
      params.detectionMinHeight = itr->second.as<gint>();
    }
    else if (paramKey == "detected-max-w"){
      params.detectionMaxWidth = itr->second.as<gint>();
    }
    else if (paramKey == "detected-max-h"){
      params.detectionMaxHeight = itr->second.as<gint>();
    }
    else if (paramKey == "minBoxes"){
      params.minBoxes = itr->second.as<gint>();
    }
    else if (paramKey == "pre-cluster-threshold"){
      params.preClusterThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "post-cluster-threshold"){
      params.postClusterThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "eps"){
      params.eps = itr->second.as<gfloat>();
    }
    else if (paramKey == "group-threshold"){
      params.groupThreshold = itr->second.as<gint>();
    }
    else if (paramKey == "min-score"){
      params.minScore = itr->second.as<gfloat>();
    }
    else if (paramKey == "dbscan-min-score"){
      params.minScore = itr->second.as<gfloat>();
    }
    else if (paramKey == "nms-iou-threshold"){
      params.nmsIOUThreshold = itr->second.as<gfloat>();
    }
    else if (paramKey == "topk"){
      params.topK = itr->second.as<gint>();
    }
    else if (paramKey == "roi-top-offset"){
      params.roiTopOffset = itr->second.as<gint>();
    }
    else if (paramKey == "roi-bottom-offset"){
      params.roiBottomOffset = itr->second.as<gint>();
    }
    else if (paramKey == "border-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = SplitString(values);
      if (vec.size() != 4){
         g_printerr
            ("Error: in border-color, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
         ret = false;
        goto done;
      }
      params.color_params.border_color.red = std::stod(vec[0]);
      params.color_params.border_color.green = std::stod(vec[1]);
      params.color_params.border_color.blue = std::stod(vec[2]);
      params.color_params.border_color.alpha = std::stod(vec[3]);
    }
    else if (paramKey == "bg-color") {
      std::string values = itr->second.as<std::string>();
      std::vector<std::string> vec = SplitString(values);

      if (vec.size() != 4) {
        g_printerr
            ("Error: Group bg-color, Number of Color params should be exactly 4 "
            "floats {r, g, b, a} between 0 and 1");
        ret = false;
        goto done;
      }
      params.color_params.bg_color.red = std::stod(vec[0]);
      params.color_params.bg_color.green = std::stod(vec[1]);
      params.color_params.bg_color.blue = std::stod(vec[2]);
      params.color_params.bg_color.alpha = std::stod(vec[3]);
      params.color_params.have_bg_color = TRUE;
    }
    else {
      printWarning ("Unknown parameter '%s' ",paramKey.c_str());
    }
  }
  m_detectorClassAttr[class_index] = params;
done:
  return ret;
}


bool PostProcessAlgorithm::HandleEvent (GstEvent *event)
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
               //g_print ("waiting for processq to be empty, buffers in processq = %ld\n", m_processQ.size());
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
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_ADDED)
  {
      gst_nvevent_parse_pad_added (event, &source_id);
  }
  if ((GstNvEventType)GST_EVENT_TYPE(event) == GST_NVEVENT_PAD_DELETED)
  {
      gst_nvevent_parse_pad_deleted (event, &source_id);
  }
  return true;
}


/* Deinitialize the Custom Lib context */
PostProcessAlgorithm::~PostProcessAlgorithm()
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

  if (m_initParams.perClassDetectionParams){
    delete[] m_initParams.perClassDetectionParams;
  }
  if (m_initParams.outputLayerNames){
    for (uint32_t i = 0; i < m_initParams.numOutputLayers; i++){
      g_free(m_initParams.outputLayerNames[i]);
      m_initParams.outputLayerNames[i] = NULL;
    }
    g_free (m_initParams.outputLayerNames);
  }
}

// Returns NvDsBatchMeta if present in the gstreamer buffer else NULL
NvDsBatchMeta *PostProcessAlgorithm::GetNVDS_BatchMeta (GstBuffer *buffer)
{
  gpointer state = NULL;
  GstMeta *gst_meta = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  while ((gst_meta = gst_buffer_iterate_meta(buffer, &state))) {
    if (!gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      continue;
    }
    NvDsMeta *dsmeta = (NvDsMeta *) gst_meta;

    if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
      if (batch_meta != NULL) {
        GST_WARNING("Multiple NvDsBatchMeta found on buffer %p", buffer);
      }
      batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
    }
  }
  return batch_meta;
}


/* Process Buffer */
BufferResult PostProcessAlgorithm::ProcessBuffer (GstBuffer *inbuf)
{
  GstMapInfo in_map_info = GST_MAP_INFO_INIT;

  GST_DEBUG_OBJECT (m_element, "PostProcessLib: ---> Inside %s frame_num = %d\n", __func__, m_frameNum++);

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (m_element, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return BufferResult::Buffer_Error;
  }
  gst_buffer_unmap(inbuf, &in_map_info);

  // Push buffer to process thread for further processing
  PacketInfo packetInfo;
  packetInfo.inbuf = inbuf;
  packetInfo.frame_num = m_frameNum;

  // Add custom preprocessing logic if required, here
  // Pass the buffer to output_loop for further processing and pushing to next component

  // Enable for dumping the input frame, for debugging purpose
  m_processLock.lock();
  m_processQ.push(packetInfo);
  m_processCV.notify_all();
  m_processLock.unlock();

  return BufferResult::Buffer_Async;
}



/* Output Processing Thread */
void PostProcessAlgorithm::OutputThread(void)
{
  GstFlowReturn flow_ret;
  GstBuffer *outBuffer = NULL;
  std::unique_lock<std::mutex> lk(m_processLock);
  NvDsBatchMeta *batch_meta = NULL;
  int32_t frame_cnt = 0;
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

    // Add post process algorithm logic here
    // Once buffer processing is done, push the buffer to the downstream
    // by using gst_pad_push function
    NvBufSurface *in_surf = getNvBufSurface (packetInfo.inbuf);
    batch_meta = GetNVDS_BatchMeta (packetInfo.inbuf);
    outBuffer = packetInfo.inbuf;
    nvds_set_input_system_timestamp (outBuffer, GST_ELEMENT_NAME(m_element));
    if(m_preprocessor_support)
    {
      for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
            l_user != NULL; l_user = l_user->next) {
          NvDsUserMeta *roi_user_meta = (NvDsUserMeta *) l_user->data;
          if (roi_user_meta->base_meta.meta_type != NVDS_ROI_META)
            continue;
          /* convert to roi metadata */
          NvDsRoiMeta *roi_meta =
            (NvDsRoiMeta *) roi_user_meta->user_meta_data;
          for (NvDsUserMetaList * r_user = roi_meta->roi_user_meta_list;
          r_user != NULL; r_user = r_user->next){
            NvDsUserMeta *tensor_user_meta = (NvDsUserMeta *) r_user->data;
            if (tensor_user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
              continue;
            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta =
              (NvDsInferTensorMeta *) tensor_user_meta->user_meta_data;
            /* PGIE and operate on meta->unique_id data only */
            if (meta->unique_id == m_gieUniqueId){
              for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
              }
              std::vector < NvDsInferLayerInfo >
                outputLayersInfo (meta->output_layers_info,
                    meta->output_layers_info + meta->num_output_layers);
              NvDsPostProcessFrameOutput output;
              memset (&output, 0, sizeof(output));
              if (m_Postprocessor){
                m_Postprocessor->setNetworkInfo(meta->network_info);
                m_Postprocessor->parseEachFrame(outputLayersInfo, output);
                m_Postprocessor->attachMetadata (in_surf, frame_meta->batch_id,
                    batch_meta, frame_meta, NULL, NULL,
                    output,
                    m_initParams.perClassDetectionParams,
                    m_filterOutClassIds,
                    m_gieUniqueId,
                    m_outputInstanceMask,
                    m_processMode, m_segmentationThreshold,
                    meta->maintain_aspect_ratio, roi_meta,
                    meta->symmetric_padding);
                m_Postprocessor->releaseFrameOutput (output);
              }
              else {
                GST_WARNING_OBJECT(m_element, "Post Processor not initialized for network");
              }
            }
          }
        }
      }
    }
    else
    {
      /* Iterate each frame metadata in batch */
      for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
          l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

        if (m_processMode == PROCESS_MODEL_FULL_FRAME){
          /* Iterate user metadata in frames to search PGIE's tensor metadata */
          for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
              l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
              continue;

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta =
              (NvDsInferTensorMeta *) user_meta->user_meta_data;
            //PGIE and operate on meta->unique_id data only
            if (meta->unique_id == m_gieUniqueId){
              for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
              }
              /* Parse output tensor and fill detection results into objectList. */
              std::vector < NvDsInferLayerInfo >
                outputLayersInfo (meta->output_layers_info,
                    meta->output_layers_info + meta->num_output_layers);
              NvDsPostProcessFrameOutput output;
              memset (&output, 0, sizeof(output));
              if (m_Postprocessor){
                m_Postprocessor->setNetworkInfo(meta->network_info);
                m_Postprocessor->parseEachFrame(outputLayersInfo, output);
                m_Postprocessor->attachMetadata (in_surf, frame_meta->batch_id,
                    batch_meta, frame_meta, NULL, NULL,
                    output,
                    m_initParams.perClassDetectionParams,
                    m_filterOutClassIds,
                    m_gieUniqueId,
                    m_outputInstanceMask,
                    m_processMode, m_segmentationThreshold,
                    meta->maintain_aspect_ratio, NULL,
                    meta->symmetric_padding);
                m_Postprocessor->releaseFrameOutput (output);
              }
              else {
                GST_WARNING_OBJECT(m_element, "Post Processor not initialized for network");
              }
            }
          }
        }
        else if (m_processMode == PROCESS_MODEL_OBJECTS){
          /* Iterate object metadata in frame */
          for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
              l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                l_user = l_user->next) {

              NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
              if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

              /* convert to tensor metadata */
              NvDsInferTensorMeta *meta =
                (NvDsInferTensorMeta *) user_meta->user_meta_data;

              if (meta->unique_id == m_gieUniqueId){
                m_Postprocessor->setNetworkInfo(meta->network_info);
                m_Postprocessor->prcoessMetadata(meta, frame_meta, obj_meta);
              }
            }
          }
        }
      }
    }

    nvds_set_output_system_timestamp (outBuffer, GST_ELEMENT_NAME(m_element));
    flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (m_element), outBuffer);
    GST_DEBUG_OBJECT (m_element,
    "CustomLib: %s in_surf=%p, Pushing Frame %d to downstream... Frame %d flow_ret = %d"\
    " TS=%" GST_TIME_FORMAT " \n",
            __func__, in_surf, packetInfo.frame_num, frame_cnt++,
            flow_ret,
            GST_TIME_ARGS(GST_BUFFER_PTS(outBuffer)));

    lk.lock();
    continue;
  }
  outputthread_stopped = true;
  lk.unlock();
  return;
}


