/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "ds_yml_parse.h"

#include <yaml-cpp/yaml.h>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_map>

NvDsYamlParserStatus
ds_parse_rtsp_output(GstElement * sink,
  GstRTSPServer *server, GstRTSPMediaFactory *factory,
  gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {

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
        g_object_set (G_OBJECT(server), "service", itr->second.as<guint>(), NULL);
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

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {

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

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {

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


guint
ds_parse_group_type(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {

      if (docs[i][group]["type"]) {
          val= docs[i][group]["type"].as<guint>();
          return val;
      }
    }
  }
  return 0;
}

guint
ds_parse_enc_type(gchar *cfg_file_path, const char* group)
{
  std::string paramKey = "";

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  guint val = 0;

  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {

      if (docs[i][group]["enc"]) {
          val= docs[i][group]["enc"].as<guint>();
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

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  std::vector<int> docs_indx_vec;
  std::unordered_map<std::string, int> docs_indx_umap;

  int total_docs = docs.size();
  GString *str = NULL;
  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {
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

  auto docs = YAML::LoadAllFromFile(cfg_file_path);

  int total_docs = docs.size();
  GString *str = NULL;

  g_print("total %d item\n",total_docs);
  for (int i =0; i < total_docs;i++)
  {
    if (docs[i][group] != NULL) {
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

