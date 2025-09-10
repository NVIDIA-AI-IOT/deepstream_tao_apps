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

#ifndef __POST_PROCESS_STRUCT_HPP__
#define __POST_PROCESS_STRUCT_HPP__

#include <iostream>
#include <vector>
#include <memory>
#include "gstnvdsinfer.h"


#ifdef __cplusplus
extern "C" {
#endif

#define _PATH_MAX 4096
#define _MAX_STR_LENGTH 1024
/**
 * Sets values on a @ref NvDsInferDimsCHW structure from a @ref NvDsInferDims
 * structure.
 */
#define getDimsCHWFromDims(dimsCHW,dims) \
  do { \
    (dimsCHW).c = (dims).d[0]; \
    (dimsCHW).h = (dims).d[1]; \
    (dimsCHW).w = (dims).d[2]; \
  } while (0)

#define getDimsHWCFromDims(dimsCHW,dims) \
  do { \
    (dimsCHW).h = (dims).d[0]; \
    (dimsCHW).w = (dims).d[1]; \
    (dimsCHW).c = (dims).d[2]; \
  } while (0)

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

inline const char *safeStr(const std::string &str)
{
  return str.c_str();
}

inline bool string_empty(const char* str) {
  return !str || strlen(str) == 0;
}

constexpr float DEFAULT_PRE_CLUSTER_THRESHOLD = 0.2;
constexpr float DEFAULT_POST_CLUSTER_THRESHOLD = 0.0;
constexpr float DEFAULT_EPS = 0.0;
constexpr int DEFAULT_GROUP_THRESHOLD = 0;
constexpr int DEFAULT_MIN_BOXES = 0;
constexpr float DEFAULT_DBSCAN_MIN_SCORE = 0;
constexpr float DEFAULT_NMS_IOU_THRESHOLD = 0.3;
constexpr int DEFAULT_TOP_K = -1;
constexpr bool ATHR_ENABLED = true;
constexpr float ATHR_THRESHOLD = 60.0;
constexpr int PROCESS_MODEL_FULL_FRAME = 1;
constexpr int PROCESS_MODEL_OBJECTS = 2;

/**
 * Enum for the status codes returned by NvDsPostProcessAlgorithm.
 */
typedef enum {
    /**  operation succeeded. */
    NVDSPOSTPROCESS_SUCCESS = 0,
    /** Failed to configure the  instance possibly due to an
     *  erroneous initialization property. */
    NVDSPOSTPROCESS_CONFIG_FAILED,
    /** Custom Library interface implementation failed. */
    NVDSPOSTPROCESS_CUSTOM_LIB_FAILED,
    /** Invalid parameters were supplied. */
    NVDSPOSTPROCESS_INVALID_PARAMS,
    /** Output parsing failed. */
    NVDSPOSTPROCESS_OUTPUT_PARSING_FAILED,
    /** CUDA error was encountered. */
    NVDSPOSTPROCESS_CUDA_ERROR,
    /** Resource error was encountered. */
    NVDSPOSTPROCESS_RESOURCE_ERROR,
    /** Unknown error was encountered. */
    NVDSPOSTPROCESS_UNKNOWN_ERROR
} NvDsPostProcessStatus;


typedef enum {
    NVDSPOSTPROCESS_LOG_ERROR = 0,
    NVDSPOSTPROCESS_LOG_WARNING,
    NVDSPOSTPROCESS_LOG_INFO,
    NVDSPOSTPROCESS_LOG_DEBUG,
} NvDsPostProcesLogLevel;

#define printMsg(level, tag_str, fmt, ...)                                  \
    do {                                                                    \
        char* baseName = strrchr((char*)__FILE__, '/');                     \
        baseName = (baseName) ? (baseName + 1) : (char*)__FILE__;           \
        char logMsgBuffer[5 * _MAX_STR_LENGTH + 1];                             \
        snprintf(logMsgBuffer, 5 * _MAX_STR_LENGTH,                             \
            tag_str " NvDsPostProcess::%s() <%s:%d> : " fmt, \
            __func__, baseName, __LINE__, ##__VA_ARGS__);       \
        fprintf(stderr, "%s\n", logMsgBuffer);                          \
    } while (0)

#define printError(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_ERROR, "Error in", fmt, ##__VA_ARGS__); \
    } while (0)

#define printWarning(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_WARNING, "Warning from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printInfo(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_INFO, "Info from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printDebug(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_DEBUG, "DEBUG", fmt, ##__VA_ARGS__); \
    } while (0)

typedef struct
{
  unsigned int roiTopOffset;
  unsigned int roiBottomOffset;
  unsigned int detectionMinWidth;
  unsigned int detectionMinHeight;
  unsigned int detectionMaxWidth;
  unsigned int detectionMaxHeight;
} NvDsPostProcessDetectionFilterParams;

/**
 * Holds the bounding box coloring information for one class;
 */
typedef struct
{
  int have_border_color;
  NvOSD_ColorParams border_color;

  int have_bg_color;
  NvOSD_ColorParams bg_color;
} NvDsPostProcessColorParams;


/**
 * Enum for clustering mode for detectors
 */
typedef enum
{
    NVDSPOSTPROCESS_CLUSTER_GROUP_RECTANGLES = 0,
    NVDSPOSTPROCESS_CLUSTER_DBSCAN,
    NVDSPOSTPROCESS_CLUSTER_NMS,
    NVDSPOSTPROCESS_CLUSTER_DBSCAN_NMS_HYBRID,
    NVDSPOSTPROCESS_CLUSTER_NONE
} NvDsPostProcessClusterMode;

/**
 * Defines UFF layer orders.
 */
typedef enum {
    NvDsPostProcessTensorOrder_kNCHW,
    NvDsPostProcessTensorOrder_kNHWC,
    NvDsPostProcessTensorOrder_kNC,
} NvDsPostProcessTensorOrder;


/**
 * Defines network types.
 */
typedef enum
{
    /** Specifies a detector. Detectors find objects and their coordinates,
     and their classes in an input frame. */
    NvDsPostProcessNetworkType_Detector,
    /** Specifies a classifier. Classifiers classify an entire frame into
     one of several classes. */
    NvDsPostProcessNetworkType_Classifier,
    /** Specifies a segmentation network. A segmentation network classifies
     each pixel into one of several classes. */
    NvDsPostProcessNetworkType_Segmentation,
    /** Specifies a instance segmentation network. A instance segmentation
     network detects objects, bounding box and mask for objects, and
     their classes in an input frame */
    NvDsPostProcessNetworkType_InstanceSegmentation,
    /** Bodypose 3D */
    NvDsPostProcessNetworkType_BodyPose,
    /** Specifies other. Output layers of an "other" network are not parsed by
     NvDsPostProcessContext. This is useful for networks that produce custom output.
     Output can be parsed by the NvDsPostProcessContext client or can be combined
     with the Gst-nvinfer feature to flow output tensors as metadata. */
    NvDsPostProcessNetworkType_Other = 100
} NvDsPostProcessNetworkType;

/**
 * Holds detection and bounding box grouping parameters.
 */
typedef struct
{
    /** Holds the bounding box detection threshold to be applied prior
     * to clustering operation. */
    float preClusterThreshold;

    /** Hold the bounding box detection threshold to be applied post
     * clustering operation. */
    float postClusterThreshold;

    /** Holds the epsilon to control merging of overlapping boxes. Refer to OpenCV
     * groupRectangles and DBSCAN documentation for more information on epsilon. */
    float eps;
    /** Holds the minimum number of boxes in a cluster to be considered
     an object during grouping using DBSCAN. */
    int minBoxes;
    /** Holds the minimum number boxes in a cluster to be considered
     an object during grouping using OpenCV groupRectangles. */
    int groupThreshold;
    /** Minimum score in a cluster for the cluster to be considered an object
     during grouping. Different clustering may cause the algorithm
     to use different scores. */
    float minScore;
    /** IOU threshold to be used with NMS mode of clustering. */
    float nmsIOUThreshold;
    /** Number of objects with objects to be filtered in the decensding order
     * of probability */
    int topK;

    unsigned int roiTopOffset;
    unsigned int roiBottomOffset;
    unsigned int detectionMinWidth;
    unsigned int detectionMinHeight;
    unsigned int detectionMaxWidth;
    unsigned int detectionMaxHeight;
    NvDsPostProcessColorParams color_params;

} NvDsPostProcessDetectionParams;



/**
 * Holds the initialization parameters required for the NvDsPostProcessContext interface.
 */
typedef struct _NvDsPostProcessContextInitParams
{
    /** Holds a unique identifier for the instance. This can be used
     to identify the instance that is generating log and error messages. */
    unsigned int uniqueID;

    /** Holds the maximum number of frames to be inferred together in a batch.
     The number of input frames in a batch must be
     less than or equal to this. */
    unsigned int maxBatchSize;

    /** Holds the pathname of the labels file containing strings for the class
     labels. The labels file is optional. The file format is described in the
     custom models section of the DeepStream SDK documentation. */
    char labelsFilePath[_PATH_MAX];


    /** Holds the network type. */
    NvDsPostProcessNetworkType networkType;

    /** Holds the number of classes detected by a detector network. */
    unsigned int numDetectedClasses;

    /** Holds per-class detection parameters. The array's size must be equal
     to @a numDetectedClasses. */
    NvDsPostProcessDetectionParams *perClassDetectionParams;

    /** Holds the minimum confidence threshold for the classifier to consider
     a label valid. */
    float classifierThreshold;

    float segmentationThreshold;

    /** Holds a pointer to an array of pointers to output layer names. */
    char ** outputLayerNames;
    /** Holds the number of output layer names. */
    unsigned int numOutputLayers;

    /** Holds the ID of the GPU which is to run the inference. */
    unsigned int gpuID;

    /** Inference input dimensions for runtime engine */
    NvDsInferDimsCHW inferInputDims;

    /** Holds the type of clustering mode */
    NvDsPostProcessClusterMode clusterMode;

    /** Holds the name of the bounding box and instance mask parse function
     in the custom library. */
    char customBBoxInstanceMaskParseFuncName[_MAX_STR_LENGTH];

      /** Holds the name of the custom bounding box function
     in the custom library. */
    char customBBoxParseFuncName[_MAX_STR_LENGTH];
    /** Name of the custom classifier attribute parsing function in the custom
     *  library. */
    char customClassifierParseFuncName[_MAX_STR_LENGTH];

    /** Holds output order for segmentation network */
    NvDsPostProcessTensorOrder segmentationOutputOrder;

    char *classifier_type;
    /** Holds boolean value to show whether preprocessor support is there. */
    gboolean preprocessor_support = FALSE;
} NvDsPostProcessContextInitParams;

/**
 * Holds information about one parsed object from a detector's output.
 */
typedef struct
{
  /** Holds the ID of the class to which the object belongs. */
  unsigned int classId;

  /** Holds the horizontal offset of the bounding box shape for the object. */
  float left;
  /** Holds the vertical offset of the object's bounding box. */
  float top;
  /** Holds the width of the object's bounding box. */
  float width;
  /** Holds the height of the object's bounding box. */
  float height;

  /** Holds the object detection confidence level; must in the range
   [0.0,1.0]. */
  float detectionConfidence;
} NvDsPostProcessObjectDetectionInfo;

/**
 * Holds information about one classified attribute.
 */
typedef struct
{
    /** Holds the index of the attribute's label. This index corresponds to
     the order of output layers specified in the @a outputCoverageLayerNames
     vector during initialization. */
    unsigned int attributeIndex;
    /** Holds the the attribute's output value. */
    unsigned int attributeValue;
    /** Holds the attribute's confidence level. */
    float attributeConfidence;
    /** Holds a pointer to a string containing the attribute's label.
     Memory for the string must not be freed. Custom parsing functions must
     allocate strings on heap using strdup or equivalent. */
    char *attributeLabel;
} NvDsPostProcessAttribute;


/**
 * A typedef defined to maintain backward compatibility.
 */
typedef NvDsPostProcessObjectDetectionInfo NvDsPostProcessParseObjectInfo;

/**
 * Holds information about one parsed object and instance mask from a detector's output.
 */
typedef struct
{
  /** Holds the ID of the class to which the object belongs. */
  unsigned int classId;

  /** Holds the horizontal offset of the bounding box shape for the object. */
  float left;
  /** Holds the vertical offset of the object's bounding box. */
  float top;
  /** Holds the width of the object's bounding box. */
  float width;
  /** Holds the height of the object's bounding box. */
  float height;

  /** Holds the object detection confidence level; must in the range
   [0.0,1.0]. */
  float detectionConfidence;

  /** Holds object segment mask */
  float *mask;
  /** Holds width of mask */
  unsigned int mask_width;
  /** Holds height of mask */
  unsigned int mask_height;
  /** Holds size of mask in bytes*/
  unsigned int mask_size;
} NvDsPostProcessInstanceMaskInfo;

/**
 * Holds information about one detected object.
 */
typedef struct
{
    /** Holds the object's offset from the left boundary of the frame. */
    float left;
    /** Holds the object's offset from the top boundary of the frame. */
    float top;
    /** Holds the object's width. */
    float width;
    /** Holds the object's height. */
    float height;
    /** Holds the index for the object's class. */
    int classIndex;
    /** Holds a pointer to a string containing a label for the object. */
    char *label;
    /* confidence score of the detected object. */
    float confidence;
    /* Instance mask information for the object. */
    float *mask;
    /** Holds width of mask */
    unsigned int mask_width;
    /** Holds height of mask */
    unsigned int mask_height;
    /** Holds size of mask in bytes*/
    unsigned int mask_size;
} NvDsPostProcessObject;

/**
 * Holds information on all objects detected by a detector network in one
 * frame.
 */
typedef struct
{
    /** Holds a pointer to an array of objects. */
    NvDsPostProcessObject *objects;
    /** Holds the number of objects in @a objects. */
    unsigned int numObjects;
} NvDsPostProcessDetectionOutput;

/**
 * Holds information on all attributes classifed by a classifier network for
 * one frame.
 */
typedef struct
{
    /** Holds a pointer to an array of attributes. There may be more than
     one attribute, depending on the number of output coverage layers
     (multi-label classifiers). */
    NvDsPostProcessAttribute *attributes;
    /** Holds the size of the @a attributes array. */
    unsigned int numAttributes;
    /** Holds a pointer to a string containing a label for the
     classified output. */
    char *label;
} NvDsPostProcessClassificationOutput;

/**
 * Holds information parsed from segmentation network output for one frame.
 */
typedef struct
{
    /** Holds the width of the output. Same as network width. */
    unsigned int width;
    /** Holds the height of the output. Same as network height. */
    unsigned int height;
    /** Holds the number of classes supported by the network. */
    unsigned int classes;
    /** Holds a pointer to an array for the 2D pixel class map.
     The output for pixel (x,y) is at index (y*width+x). */
    int *class_map;
    /** Holds a pointer to an array containing raw probabilities.
     The probability for class @a c and pixel (x,y) is at index
     (c*width*height + y*width+x). */
    float *class_probability_map;
} NvDsPostProcessSegmentationOutput;

typedef struct NvDsPoint3f {
  float x;
  float y;
  float z;
} NvDsPoint3f;

/**
 * Holds information parsed from bodypose network output for one frame.
 */
typedef struct
{
    /** Holds the width of the output. Same as network width. */
    unsigned int width;
    /** Holds the height of the output. Same as network height. */
    unsigned int height;

    unsigned int num_key_points;
    float *data;
} NvDsPostProcessBodyPoseOutput;


/**
 * Holds the information inferred by the network on one frame.
 */
typedef struct
{
    /** Holds an output type indicating the valid member in the union
     of @a detectionOutput, @a classificationOutput, and @a  segmentationOutput.
     This is basically the network type. */
    NvDsPostProcessNetworkType outputType;
    /** Holds a union of supported outputs. The valid member is determined by
     @a outputType. */
    union
    {
        /** Holds detector output. Valid when @a outputType is
         @ref NvDsPostProcessNetworkType_Detector. */
        NvDsPostProcessDetectionOutput detectionOutput;
        /** Holds classifier output. Valid when @a outputType is
         @ref NvDsPostProcessNetworkType_Classifier. */
        NvDsPostProcessClassificationOutput classificationOutput;
        /** Holds classifier output. Valid when @a outputType is
         @ref NvDsPostProcessNetworkType_Classifier. */
        NvDsPostProcessSegmentationOutput segmentationOutput;
        /** Holds classifier output. Valid when @a outputType is
         @ref NvDsPostProcessNetworkType_Classifier. */
        NvDsPostProcessBodyPoseOutput bodyPoseOutput;
    };
} NvDsPostProcessFrameOutput;

/**
 * Holds the output for all of the frames in a batch (an array of frame),
 * and related buffer information.
 */
typedef struct
{
    /** Holds a pointer to an array of outputs for each frame in the batch. */
    NvDsPostProcessFrameOutput *frames;
    /** Holds the number of elements in @a frames. */
    unsigned int numFrames;

    /** Holds a pointer to an array of pointers to output device buffers
     for this batch. The array elements are set by */
    void **outputDeviceBuffers;
    /** Holds the number of elements in @a *outputDeviceBuffers. */
    unsigned int numOutputDeviceBuffers;

    /** Holds a pointer to an array of pointers to host buffers for this batch.
     The array elements are set by */
    void **hostBuffers;
    /** Holds the number of elements in hostBuffers. */
    unsigned int numHostBuffers;

    /** Holds a private context pointer for the set of output buffers. */
    void* priv;
} NvDsPostProcessBatchOutput;

#ifdef __cplusplus
}
#endif

/**
 * Holds the detection parameters required for parsing objects.
 */
typedef struct
{
  /** Holds the number of classes requested to be parsed, starting with
   class ID 0. Parsing functions may only output objects with
   class ID less than this value. */
  unsigned int numClassesConfigured;
  /** Holds a per-class vector of detection confidence thresholds
   to be applied prior to the clustering operation.
   Parsing functions may only output an object with detection confidence
   greater than or equal to the vector element indexed by the object's
   class ID. */
  std::vector<float> perClassPreclusterThreshold;
  /* Per-class threshold to be applied after the clustering operation. */
  std::vector<float> perClassPostclusterThreshold;

} NvDsPostProcessParseDetectionParams;


/** Holds the cached information of an object. */
struct NvDsPostProcessObjectInfo {
  /** Vector of cached classification attributes. */
  std::vector<NvDsPostProcessAttribute> attributes;
  /** Cached string label. */
  std::string label;

  NvDsPostProcessObjectInfo(const NvDsPostProcessObjectInfo&) = delete;
  NvDsPostProcessObjectInfo() = default;
  ~NvDsPostProcessObjectInfo(){
    for (auto &attr : attributes) {
      if (attr.attributeLabel)
        free (attr.attributeLabel);
    }
  }
};


/**
 * Holds the inference information/history for one object based on it's
 * tracking id.
 */
typedef struct _NvDsPostProcessObjectHistory
{
  /** Boolean indicating if the object is already being inferred on. */
  int under_inference;
  /** Bounding box co-ordinates of the object when it was last inferred on. */
  NvOSD_RectParams last_inferred_coords;
  /** Number of the frame in the stream when the object was last inferred on. */
  unsigned long last_inferred_frame_num;
  /** Number of the frame in the stream when the object was last accessed. This
   * is useful for clearing stale enteries in map of the object histories and
   * keeping the size of the map in check. */
  unsigned long last_accessed_frame_num;
  /** Cached object information. */
  NvDsPostProcessObjectInfo cached_info;
} NvDsPostProcessObjectHistory;


#endif
