/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __POSTPROCESSLIB_HPP__
#define __POSTPROCESSLIB_HPP__

#include <cuda.h>
#include <cuda_runtime.h>
#include "post_processor.h"
#include "post_processor_bodypose.h"
#include "nvdspostprocesslib_base.hpp"

#define FORMAT_NV12 "NV12"
#define FORMAT_RGBA "RGBA"
#define FORMAT_I420 "I420"
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

/* Strcture used to share between the threads */
struct PacketInfo {
  GstBuffer *inbuf;
  guint frame_num;
};

class PostProcessAlgorithm : public DSPostProcessLibraryBase
{
public:

  explicit PostProcessAlgorithm(DSPostProcess_CreateParams *createParams) {
    m_vectorProperty.clear();
    outputthread_stopped = false;

    if (createParams){
      m_element = createParams->m_element;
      m_gpuId = createParams->m_gpuId;
      m_cudaStream = createParams->m_cudaStream;
      m_preprocessor_support = createParams->m_preprocessor_support;
    }
    else {
      m_element = NULL;
      m_gpuId = 0;
      m_cudaStream = 0;
      m_preprocessor_support = FALSE;
    }
    m_initParams.preprocessor_support = FALSE;
    m_outputThread = new std::thread(&PostProcessAlgorithm::OutputThread, this);
    m_initParams.uniqueID = 0;
    m_initParams.maxBatchSize = 1;

    std::memset(m_initParams.labelsFilePath,0, sizeof(m_initParams.labelsFilePath));
    m_initParams.networkType = NvDsPostProcessNetworkType_Other;
    std::memset(m_initParams.customClassifierParseFuncName, 0,
        sizeof(m_initParams.customClassifierParseFuncName)-1);
    std::memset(m_initParams.customBBoxParseFuncName, 0,
        sizeof(m_initParams.customBBoxParseFuncName)-1);
    std::memset(m_initParams.customBBoxInstanceMaskParseFuncName, 0,
        sizeof(m_initParams.customBBoxInstanceMaskParseFuncName)-1);


    /** Holds the number of classes detected by a detector network. */
    m_initParams.numDetectedClasses = 0;

    /** Holds per-class detection parameters. The array's size must be equal
     to @a numDetectedClasses. */
    m_initParams.perClassDetectionParams = NULL;

    /** Holds the minimum confidence threshold for the classifier to consider
     a label valid. */
    m_initParams.classifierThreshold = 0.5;

    m_initParams.segmentationThreshold = 0.5;

    /** Holds a pointer to an array of pointers to output layer names. */
    m_initParams.outputLayerNames = NULL;
    /** Holds the number of output layer names. */
    m_initParams.numOutputLayers = 0;

    /** Holds the ID of the GPU which is to run the inference. */
    m_initParams.gpuID = 0;

    m_initParams.inferInputDims = {0,0,0};

    /** Holds the type of clustering mode */
    m_initParams.clusterMode = NVDSPOSTPROCESS_CLUSTER_NMS;

    m_initParams.classifier_type = NULL;
  }

  /* Pass GST events to the library */
  virtual bool HandleEvent(GstEvent *event);

  /* Read Config file */
  virtual bool SetConfigFile (const gchar *config_file);

  /* Process Incoming Buffer */
  virtual BufferResult ProcessBuffer(GstBuffer *inbuf);

  std::vector<std::string> SplitString (std::string input);

  std::set<gint> SplitStringInt (std::string input);

  bool GetAbsFilePath (const gchar * cfg_file_path, const gchar * file_path,
    char *abs_path_str);

  gboolean hw_caps;

  /* Deinit members */
  ~PostProcessAlgorithm();

private:
  /* Helper Function to Extract Batch Meta from buffer */
  NvDsBatchMeta * GetNVDS_BatchMeta (GstBuffer *buffer);

  /* Output Processing Thread, push buffer to downstream  */
  void OutputThread(void);

  /* Helper function to Dump NvBufSurface RAW content */
  void DumpNvBufSurface (NvBufSurface *in_surface, NvDsBatchMeta *batch_meta);

  bool ParseLabelsFile(const std::string& labelsFilePath);
  bool ParseConfAttr (YAML::Node node, gint64 class_id, NvDsPostProcessDetectionParams& params);
  NvDsPostProcessStatus preparePostProcess();

  NvDsPostProcessNetworkType m_networkType;
  NvDsPostProcessClusterMode m_clusterMode;
  gfloat m_classifierThreshold;
  gfloat m_segmentationThreshold;
  gint m_numDetectedClasses;
  gint m_processMode = 1;
  guint m_gieUniqueId = 0;
  gboolean m_isClassifier = 0;
  gboolean m_releaseTensorMeta = FALSE;
  gboolean m_outputInstanceMask = FALSE;
  gboolean m_preprocessor_support = FALSE;
  std::string m_classifierType;
  std::set <gint> m_filterOutClassIds;
  std::set <gint> m_operateOnClassIds;
  std::vector <std::string> m_outputBlobNames;

  std::vector<NvDsPostProcessInstanceMaskInfo> m_InstanceMaskList;
  std::unordered_map <gint, NvDsPostProcessDetectionParams> m_detectorClassAttr;
  /* Vector of NvDsPostProcessInstanceMaskInfo vectors for each class. */
  std::vector<std::vector<NvDsPostProcessInstanceMaskInfo>> m_PerClassInstanceMaskList;
  std::vector<std::vector<std::string>> m_Labels;

  std::unique_ptr<ModelPostProcessor> m_Postprocessor;
  NvDsPostProcessContextInitParams m_initParams;
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

};

extern "C" IDSPostProcessLibrary *CreateCustomAlgoCtx(DSPostProcess_CreateParams *params);
// Create Custom Algorithm / Library Context


#endif
