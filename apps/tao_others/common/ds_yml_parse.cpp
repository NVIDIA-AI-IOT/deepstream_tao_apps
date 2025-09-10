/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <yaml-cpp/yaml.h>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include "ds_yml_parse.h"
#include "nvds_yml_parser.h"

static gchar *get_absolute_file_path(gchar *cfg_file_path, const gchar *file_path) {
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return (gchar *)file_path;
  }

  if (!realpath(cfg_file_path, abs_cfg_path)) {
    return NULL;
  }

  /* Return absolute path of config file if file_path is NULL. */
  if (!file_path) {
    abs_file_path = g_strdup(abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr(abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat(abs_cfg_path, file_path, NULL);
  return abs_file_path;
}

NvDsYamlParserStatus
ds_parse_rtsp_output(GstElement * sink,
  GstRTSPServer *server, GstRTSPMediaFactory *factory,
  gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);

    }
  }
  
  if (!sink || !server || !factory)
  {
      std::cerr << "[ERROR] Pass element does not exist!" << std::endl;
      return NVDS_YAML_PARSER_ERROR;
  } else {
    GstElementFactory *factory = GST_ELEMENT_GET_CLASS(sink)->elementfactory;
    if (g_strcmp0(GST_OBJECT_NAME(factory), "udpsink")) {
      std::cerr << "[ERROR] Passed element is not udpsink" << std::endl;
      return NVDS_YAML_PARSER_ERROR;
    }
  }

  int docs_indx_vec_size = docs_indx_vec.size();

  int docs_indx_umap_size = docs_indx_umap.size();

  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::cerr << "[ERROR] Duplicate group names in the config file : " << group << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr)
    {
      paramKey = itr->first.as<std::string>();

      if(paramKey == "udpport") {
        char udpsrc_pipeline[512];
        g_object_set (G_OBJECT (sink), "host", "224.224.255.255", "port",
        itr->second.as<guint>(), "async", FALSE, "sync", 0, NULL);
        sprintf (udpsrc_pipeline,
          "( udpsrc name=pay0 port=%s buffer-size=65536 caps=\"application/x-rtp, media=video, "
          "clock-rate=90000, encoding-name=H264, payload=96 \" )",
          itr->second.as<std::string>().c_str());
        gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);
      }
      else if(paramKey == "rtspport") {
        g_object_set (G_OBJECT(server), "service", itr->second.as<std::string>().c_str(), NULL);
        g_print("Please reach RTSP with rtsp://ip:%s/ds-out-avc\n", itr->second.as<std::string>().c_str());
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found : " << paramKey << std::endl;
      }
    }
  }

  return NVDS_YAML_PARSER_SUCCESS;
}

NvDsYamlParserStatus
ds_parse_enc_config(GstElement *encoder, 
  gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);

    }
  }

  if (!encoder)
    return NVDS_YAML_PARSER_ERROR;
  else {
    GstElementFactory *factory = GST_ELEMENT_GET_CLASS(encoder)->elementfactory;
    if (g_strcmp0(GST_OBJECT_NAME(factory), "nvv4l2h264enc")
      && g_strcmp0(GST_OBJECT_NAME(factory), "nvv4l2h265enc")) {
      std::cerr << "[ERROR] Passed element is not encoder" << std::endl;
      return NVDS_YAML_PARSER_ERROR;
    }
  }

  int docs_indx_vec_size = docs_indx_vec.size();

  int docs_indx_umap_size = docs_indx_umap.size();

  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::cerr << "[ERROR] Duplicate group names in the config file : " << group << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr)
    {
      paramKey = itr->first.as<std::string>();

      if(paramKey == "bitrate") {
        g_object_set(G_OBJECT(encoder), "bitrate",
                     itr->second.as<guint>(), NULL);
      }
      else if(paramKey == "iframeinterval") {
        g_object_set(G_OBJECT(encoder), "iframeinterval",
                     itr->second.as<guint>(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found : " << paramKey << std::endl;
      }
    }
  }

  return NVDS_YAML_PARSER_SUCCESS;
}

NvDsYamlParserStatus
ds_parse_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);

    }
  }

  if (!vtemplate)
    return NVDS_YAML_PARSER_ERROR;
  else {
    GstElementFactory *factory = GST_ELEMENT_GET_CLASS(vtemplate)->elementfactory;
    if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdsvideotemplate")) {
      std::cerr << "[ERROR] Passed element is not nvdsvideotemplate" << std::endl;
      return NVDS_YAML_PARSER_ERROR;
    }
  }

  int docs_indx_vec_size = docs_indx_vec.size();

  int docs_indx_umap_size = docs_indx_umap.size();

  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::cerr << "[ERROR] Duplicate group names in the config file : " << group << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr)
    {
      paramKey = itr->first.as<std::string>();

      if(paramKey == "customlib-name") {
        g_object_set(G_OBJECT(vtemplate), "customlib-name",
                     itr->second.as<std::string>().c_str(), NULL);
      }
      else if(paramKey == "customlib-props") {
        g_object_set(G_OBJECT(vtemplate), "customlib-props",
                     itr->second.as<std::string>().c_str(), NULL);
      }
      else {
        std::cerr << "!! [WARNING] Unknown param found : " << paramKey << std::endl;
      }
    }
  }

  return NVDS_YAML_PARSER_SUCCESS;
}

NvDsYamlParserStatus
ds_parse_ocdr_videotemplate_config(GstElement *vtemplate, 
  gchar *cfg_file_path, const char* group)
{
  if (!vtemplate)
    return NVDS_YAML_PARSER_ERROR;
  else {
    GstElementFactory *factory = GST_ELEMENT_GET_CLASS(vtemplate)->elementfactory;
    if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdsvideotemplate")) {
      std::cerr << "[ERROR] Passed element is not nvdsvideotemplate" << std::endl;
      return NVDS_YAML_PARSER_ERROR;
    }
  }

  YAML::Node node = YAML::LoadFile(cfg_file_path);

  YAML::Node docs = node[group];

  if(docs["customlib-name"]) {
    std::string libPath = docs["customlib-name"].as<std::string>();
    g_object_set(G_OBJECT(vtemplate), "customlib-name",
                 libPath.c_str(), NULL);
  }

  if(docs["customlib-props"]) {
    auto listNode = docs["customlib-props"];
    for(uint32_t i = 0; i < listNode.size(); i++) {
      std::string tmpProb = listNode[i].as<std::string>();
      g_object_set(G_OBJECT(vtemplate), "customlib-props",
                   tmpProb.c_str(), NULL);
    }
  }

  return NVDS_YAML_PARSER_SUCCESS;
}


guint
ds_parse_group_type(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {

      if (docs[i][group]["type"]) {
          val= docs[i][group]["type"].as<guint>();
          return val;
      }
    }
  }
  return 0;
}

NvDsYamlParserStatus ds_parse_nvdsanalytics(GstElement *element, gchar *cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;
  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdsanalytics")) {
    std::cerr << "[ERROR] Passed element is not nvdsanalytics" << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  std::string paramKey = "";
  auto docs = YAML::LoadAllFromFile(cfg_file_path);
  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;
  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group].Type() != YAML::NodeType::Null) {
      if (docs[i][group]["enable"]) {
        gboolean val= docs[i][group]["enable"].as<gboolean>();
        if(val == FALSE) {
          g_print("!! [WARNING]  \"analytics\" group not enabled.\n");
          g_object_set(G_OBJECT(element), "enable", val, NULL);
        }
      }

      YAML::const_iterator itr = docs[i].begin();
      std::string group_name = itr->first.as<std::string>();
      docs_indx_umap[group_name] = i;
      docs_indx_vec.push_back(i);
    }
  }

  int docs_indx_vec_size = docs_indx_vec.size();
  int docs_indx_umap_size = docs_indx_umap.size();
  if (docs_indx_umap_size != docs_indx_vec_size) {
    std::cerr << "[ERROR] Duplicate group names in the config file : " << group << std::endl;
    ret = NVDS_YAML_PARSER_ERROR;
    return ret;
  }

  for (int i = 0; i< docs_indx_vec_size; i++)
  {
    int indx = docs_indx_vec [i];
    for(YAML::const_iterator itr = docs[indx][group].begin(); itr != docs[indx][group].end(); ++itr)
    {
      paramKey = itr->first.as<std::string>();

      if (paramKey == "enable" && docs[indx][group]["enable"].as<gboolean>() == TRUE) {
        continue;
      } else if(paramKey == "config-file") {
        std::string temp = itr->second.as<std::string>();
        if (temp.empty()) {
          g_printerr ("Error: Could not parse config-file-path in %s.\n", group);
          return NVDS_YAML_PARSER_ERROR;
        }
        char *config_file_path = get_absolute_file_path(cfg_file_path, temp.c_str());
        if (!config_file_path) {
          g_printerr ("Error: Could not get absolute path for config-file-path in %s.\n", group);
          return NVDS_YAML_PARSER_ERROR;
        }
        g_print("Setting config-file for nvdsanalytics: %s\n", config_file_path);
        g_object_set(G_OBJECT(element), "config-file", config_file_path, NULL);
        g_free(config_file_path);
      } else {
        std::cerr << "!! [WARNING] Unknown param found for nvdsanalytics: " << paramKey << std::endl;
      }
    }
  }

  return ret;
}

static const char* dgpus_unsupport_hw_enc[] = {
  "NVIDIA A100",
  "NVIDIA A30",
  "NVIDIA H100", // NVIDIA H100 SXM, NVIDIA H100 PCIe, NVIDIA H100 NVL
  "NVIDIA T500",
  "GeForce MX570 A",
  "DGX A100"
};

static bool
is_enc_hw_support() {
  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  bool enc_hw_support = TRUE;
  if (prop.integrated) {
    char device_name[50];
    FILE* ptr = fopen("/proc/device-tree/model", "r");

    if (ptr) {
      while (fgets(device_name, 50, ptr) != NULL) {
        if (strstr(device_name,"Orin") && (strstr(device_name,"Nano")))
          enc_hw_support = FALSE;
        }
    }
    fclose(ptr);
  } else {
    for (uint32_t i = 0; i < sizeof(dgpus_unsupport_hw_enc)/sizeof(dgpus_unsupport_hw_enc[0]); i++) {
      if (!strncasecmp(prop.name, dgpus_unsupport_hw_enc[i], strlen(dgpus_unsupport_hw_enc[i]))) {
        enc_hw_support = FALSE;
        break;
      }
    }
  }
  return enc_hw_support;
}

guint
ds_parse_enc_type(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {
      if (docs[i][group]["enc-type"]) {
        val = docs[i][group]["enc-type"].as<guint>();
        // If hardware encoding is configured but the hardware does not support it, 
        // fallback to software encoding
        if (val == 0 && !is_enc_hw_support()) {
          g_print("** WARN: hardware encoding is not supported, fallback to software encoding \n");
          val = 1;
        }
        return val;
      }
    }
  }
  return 0;
}

guint
ds_parse_enc_codec(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {

      if (docs[i][group]["codec"]) {
          val= docs[i][group]["codec"].as<guint>();
          return val;
      }
    }
  }
  return 0;
}

GString *
ds_parse_file_name(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  GString *str = NULL;
  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {
      if (docs[i][group]["filename"]) {
          std::string temp = docs[i][group]["filename"].as<std::string>();
          str = g_string_new(temp.c_str());
          return str;
      }
    }
  }
  return NULL;
}

GString *
ds_parse_config_yml_filepath(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);

  int total_docs = docs.size();
  GString *str = NULL;

  g_print("total %d item\n",total_docs);
  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {
      g_print("group %s found %d\n", group, !(docs[i][group]["config-file-path"]));
      if (docs[i][group]["config-file-path"]) {
          std::string temp = docs[i][group]["config-file-path"].as<std::string>();
          str = g_string_new(temp.c_str());
          return str;
      }
    }
  }
  return NULL;
}

/* this is only for video encoding. */
void
create_video_encoder(bool isH264, int enc_type, GstElement** conv_capfilter,
  GstElement** outenc, GstElement** encparse, GstElement** rtppay)
{
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;

  g_print("in create_video_encoder, isH264:%d, enc_type:%d\n", isH264, enc_type);
  caps =  gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING,
          "I420", NULL);
  if(enc_type == ENCODER_TYPE_HW) {
    feature = gst_caps_features_new ("memory:NVMM", NULL);
    gst_caps_set_features (caps, 0, feature);
  }
  g_object_set (G_OBJECT (*conv_capfilter), "caps", caps, NULL);

  if(isH264) {
    if(enc_type == ENCODER_TYPE_HW) {
      *outenc = gst_element_factory_make ("nvv4l2h264enc" ,"nvvideo-h264enc");
    } else {
      *outenc = gst_element_factory_make ("x264enc" ,"x264enc");
    }
    *encparse = gst_element_factory_make ("h264parse", "encparse");
    if(rtppay)
      *rtppay = gst_element_factory_make ("rtph264pay", "rtppay");
  } else {
    if(enc_type == ENCODER_TYPE_HW) {
      *outenc = gst_element_factory_make ("nvv4l2h265enc" ,"nvvideo-h265enc");
    } else {
      *outenc = gst_element_factory_make ("x265enc" ,"x265enc");
    }
    *encparse = gst_element_factory_make ("h265parse", "encparse");
    if(rtppay)
      *rtppay = gst_element_factory_make ("rtph265pay", "rtppay");
  }
}

guint
ds_parse_group_enable(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";
  std::vector<YAML::Node> docs = YAML::LoadAllFromFile(cfg_file_path);
  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;
  int total_docs = docs.size();
  guint val = 0;
  for (int i =0; i < total_docs;i++)
  {
    if (!docs[i][group].IsNull()) {
      if (docs[i][group]["enable"]) {
          val= docs[i][group]["enable"].as<guint>();
          return val;
      }
    }
  }
  return 0;
}

/** Function to get the absolute path of a file.*/
gboolean
get_absolute_file_path_yaml (
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
    else
      return FALSE;
  }

  g_free (abs_file_path);

  g_strlcpy (abs_path_str, abs_real_file_path, _PATH_MAX);
  return TRUE;
}

NvDsYamlParserStatus
nvds_parse_postprocess (GstElement *element, gchar* app_cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;
  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdspostprocess")) {
    std::cerr << "[ERROR] Passed element is not nvdspostprocess" << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  if (!app_cfg_file_path) {
    printf("Config file not provided.\n");
    return NVDS_YAML_PARSER_ERROR;
  }

  YAML::Node configyml = YAML::LoadFile(app_cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      char *path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            path)) {
            ret = NVDS_YAML_PARSER_ERROR;
      }
      g_object_set(G_OBJECT(element), "postprocesslib-config-file",
         path, NULL);
         printf("postprocesslib-config-file:%s\n", path);
      g_free (str);
      g_free(path);
    } else if (paramKey == "lib-name") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      char *path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            path)) {
            ret = NVDS_YAML_PARSER_ERROR;
      }
      g_object_set(G_OBJECT(element), "postprocesslib-name",
         path, NULL);
         printf("postprocesslib-name:%s\n", path);
      g_free (str);
      g_free(path);
    } else {
      printf("[WARNING] Unknown param found in postprocess: %s\n", paramKey.c_str());
    }
  }
  return ret;
}

NvDsYamlParserStatus
nvds_parse_preprocess (GstElement *element, gchar* app_cfg_file_path, const char* group)
{
  NvDsYamlParserStatus ret = NVDS_YAML_PARSER_SUCCESS;
  GstElementFactory *factory = GST_ELEMENT_GET_CLASS(element)->elementfactory;
  if (g_strcmp0(GST_OBJECT_NAME(factory), "nvdspreprocess")) {
    std::cerr << "[ERROR] Passed element is not nvdspreprocess" << std::endl;
    return NVDS_YAML_PARSER_ERROR;
  }

  if (!app_cfg_file_path) {
    printf("Config file not provided.\n");
    return NVDS_YAML_PARSER_ERROR;
  }

  YAML::Node configyml = YAML::LoadFile(app_cfg_file_path);
  for(YAML::const_iterator itr = configyml[group].begin();
     itr != configyml[group].end(); ++itr)
  {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "config-file-path") {
      std::string temp = itr->second.as<std::string>();
      char* str = (char*) malloc(sizeof(char) * 1024);
      std::strncpy (str, temp.c_str(), 1024);
      char *config_file_path = (char*) malloc(sizeof(char) * 1024);
      if (!get_absolute_file_path_yaml (app_cfg_file_path, str,
            config_file_path)) {
            ret = NVDS_YAML_PARSER_ERROR;
      }
       g_object_set(G_OBJECT(element), "config-file",
         config_file_path, NULL);
         printf("config_file_path:%s\n", config_file_path);
      g_free (str);
      g_free(config_file_path);
    } else {
      printf("[WARNING] Unknown param found in preprocess: %s\n", paramKey.c_str());
    }
  }
  return ret;
}

void
parse_streammux_width_height_yaml (gint *width, gint *height, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["streammux"].begin();
     itr != configyml["streammux"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "width") {
      *width = itr->second.as<gint>();
    } else if(paramKey == "height"){
      *height = itr->second.as<gint>();
    }
  }
}

void
parse_sink_type_yaml (gint *type, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["sink"].begin();
     itr != configyml["sink"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "sink-type") {
      *type = itr->second.as<gint>();
    }
  }
}

void
parse_sink_enc_type_yaml (gint *enc_type, gchar *cfg_file_path)
{
  YAML::Node configyml = YAML::LoadFile(cfg_file_path);

  for(YAML::const_iterator itr = configyml["sink"].begin();
     itr != configyml["sink"].end(); ++itr) {
    std::string paramKey = itr->first.as<std::string>();
    if (paramKey == "enc-type") {
       int value = itr->second.as<gint>();
       if(value == 0 || value == 1){
        *enc_type = value;
       }
    }
  }
}