/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <sstream>
#include <vector>
#include <array>
#include <cuda_fp16.h>

#include "nvdspreprocess_lib.h"
#include "roi_conversion.hpp"

#if defined(__aarch64__)
#include "cudaEGL.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#define NVTX_DEEPBLUE_COLOR 0xFF667EBE

/** pixel-normalization-factor config parameter */
#define NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR "pixel-normalization-factor"

/** offsets config parameter */
#define NVDSPREPROCESS_USER_CONFIGS_OFFSETS "offsets"

#define NVDSPREPROCESS_USER_CONFIGS_SCALING_FILTER "scaling-filter"

#define NVDSPREPROCESS_USER_CONFIGS_SCALE_TYPE "scale-type"

#define NVDSPREPROCESS_USER_CONFIGS_AFFINE_MATRIX "affine-matrix"

#define MAX_CACHED_EGL_FRAME_SIZE 10000

#define checkRuntime(call) check_runtime(call, #call, __LINE__, __FILE__)
bool __inline__ check_runtime(cudaError_t e, const char *call, int line, const char *file) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call,
                 cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
    return false;
  }
  return true;
}

struct CustomCtx {
  std::shared_ptr<roiconv::ROIConversion> conv_handle;
  /** Custom mean subtraction and normalization parameters */
  std::array<float, 3> scales{1.0f, 1.0f, 1.0f};
  std::array<float, 3> offsets{0.0f, 0.0f, 0.0f};
  std::array<float, 6> affine_matrix{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  enum ScaleType {
    FitXY = 0,
    FitCenter = 1,
    Matrix = 2,
  };
  ScaleType scale_type = FitCenter;

  /** interpolation filter for transformation */
  roiconv::Interpolation interpolation = roiconv::Interpolation::Nearest;
  NvDsPreProcessNetworkSize network_size;
  cudaStream_t stream;
  nvtxDomainHandle_t nvtx_domain;

#if defined(__aarch64__)
  using GraphRes =
      std::tuple<CUgraphicsResource, CUeglFrame, cudaSurfaceObject_t, cudaSurfaceObject_t>;
  using EglFrameCacheMap = std::unordered_map<const NvBufSurfaceParams *, GraphRes>;
  EglFrameCacheMap egl_frame_cache;
#endif
};

static roiconv::Interpolation to_roiconv_interpolation(const std::string &input) {
  int val = std::stoi(input);
  roiconv::Interpolation ret = roiconv::Interpolation::NoneEnum;
  printf("%d is set for scaling-filter, ", (int)ret);
  switch (val) {
    case 0:
    case 2:
      ret = roiconv::Interpolation::Nearest;
      printf("using Nearest\n");
      break;
    case 1:
      ret = roiconv::Interpolation::Bilinear;
      printf("using Bilinear\n");
      break;
    default:
      printf("unsupported value, use default(Nearest)\n");
      ret = roiconv::Interpolation::Nearest;
      break;
  }
  return ret;
}

// Currently only supports NVBUF_COLOR_FORMAT_NV12 input
// NV12BlockLinear   = 1,   // Y, UV   stride = width
// NV12PitchLinear   = 2,   // Y, UV   stride = width
static roiconv::InputFormat to_roiconv_input_format(NvBufSurfaceParams *surface_params) {
  switch (surface_params->colorFormat) {
    case NVBUF_COLOR_FORMAT_NV12:
      if (surface_params->layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
        return roiconv::InputFormat::NV12BlockLinear;
      } else if (surface_params->layout == NVBUF_LAYOUT_PITCH) {
        return roiconv::InputFormat::NV12PitchLinear;
      }
    case NVBUF_COLOR_FORMAT_RGBA:
      return roiconv::InputFormat::RGBA;
    case NVBUF_COLOR_FORMAT_YUV422:
      // return roiconv::InputFormat::YUV422Packed_YUYV;
    case NVBUF_COLOR_FORMAT_YUV420:
      // return roiconv::InputFormat::YUVI420Separated;
    case NVBUF_COLOR_FORMAT_RGB:
      // return roiconv::InputFormat::RGB;
    default:
      printf("unsupport input surface format %d \n", surface_params->colorFormat);
      return roiconv::InputFormat::NoneEnum;
  }
}

static roiconv::OutputFormat to_roiconv_output_format(NvDsPreProcessNetworkInputOrder order,
                                                      NvDsPreProcessFormat format) {
  switch (format) {
    case NvDsPreProcessFormat_RGB:
      if (order == NvDsPreProcessNetworkInputOrder_kNCHW) {
        return roiconv::OutputFormat::CHW_RGB;
      } else if (order == NvDsPreProcessNetworkInputOrder_kNHWC) {
        return roiconv::OutputFormat::HWC_RGB;
      }
    case NvDsPreProcessFormat_BGR:
      if (order == NvDsPreProcessNetworkInputOrder_kNCHW) {
        return roiconv::OutputFormat::CHW_BGR;
      } else if (order == NvDsPreProcessNetworkInputOrder_kNHWC) {
        return roiconv::OutputFormat::HWC_BGR;
      }
    case NvDsPreProcessFormat_GRAY:
      return roiconv::OutputFormat::Gray;
    default:
      printf("unknow output format %d \n", format);
      return roiconv::OutputFormat::NoneEnum;
  }
}

static roiconv::OutputDType to_roiconv_output_dtype(NvDsDataType data_type) {
  switch (data_type) {
    case NvDsDataType_UINT8:
    case NvDsDataType_INT8:
      return roiconv::OutputDType::Uint8;
    case NvDsDataType_FP32:
      return roiconv::OutputDType::Float32;
    case NvDsDataType_FP16:
      return roiconv::OutputDType::Float16;
    default:
      printf("unknow output dtype %d \n", data_type);
      return roiconv::OutputDType::NoneEnum;
  }
}

template <size_t N>
static std::array<float, N> split_string(const std::string &input, char delimiter = ';') {
  std::array<float, N> array;
  std::istringstream iss(input);
  std::string token;
  size_t index = 0;

  while (std::getline(iss, token, delimiter)) {
    // Check if we are within bounds
    if (index < N) {
      // Convert token to float and assign to array
      array[index] = std::stof(token);
      index++;
    } else {
      // Stop if we exceed the size of the array
      break;
    }
  }

  return array;
}

#if defined(__aarch64__)
using EglFrameCacheIter = CustomCtx::EglFrameCacheMap::iterator;
static void release_egl_frame_cache_entry(EglFrameCacheIter it) {
  if (std::get<2>(it->second)) {
    // y plane
    checkRuntime(cudaDestroySurfaceObject(std::get<2>(it->second)));
  }
  if (std::get<3>(it->second)) {
    // uv plane
    checkRuntime(cudaDestroySurfaceObject(std::get<3>(it->second)));
  }
  cuGraphicsUnregisterResource(std::get<0>(it->second));
}

static std::pair<EglFrameCacheIter, bool> cache_egl_frame(CustomCtx *ctx, NvBufSurface *surface,
                                                          int batch_index) {
  const NvBufSurfaceParams *surface_params = &surface->surfaceList[batch_index];
  CUresult ret;
  cudaSurfaceObject_t y_plane = 0;
  cudaSurfaceObject_t uv_plane = 0;

  NvBufSurfaceMapEglImage(surface, batch_index);

  // EGLImage is in GPU memory address
  EGLImageKHR eglimage = surface_params->mappedAddr.eglImage;
  CUgraphicsResource resource;
  EglFrameCacheIter null_iter;
  // Register EGLImage to graphics resource
  ret = cuGraphicsEGLRegisterImage(&resource, eglimage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
  if (ret != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed\n");
    return std::make_pair(null_iter, false);
  }
  // Map to CUeglFrame, for direct CUDA use
  CUeglFrame eglFrame;
  ret = cuGraphicsResourceGetMappedEglFrame(&eglFrame, resource, 0, 0);
  if (ret != CUDA_SUCCESS) {
    printf("cuGraphicsResourceGetMappedEglFrame failed\n");
    return std::make_pair(null_iter, false);
  }

  if (surface_params->layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
    void *ptr0 = eglFrame.frame.pArray[0];
    void *ptr1 = eglFrame.frame.pArray[1];
    // Create the surface objects for Y
    cudaResourceDesc resDesc;

    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = (cudaArray_t)ptr0;
    checkRuntime(cudaCreateSurfaceObject(&y_plane, &resDesc));

    // Create the surface objects for UV
    resDesc.res.array.array = (cudaArray_t)ptr1;
    checkRuntime(cudaCreateSurfaceObject(&uv_plane, &resDesc));
  }

  ///\ If the saved egl mapped entries exceed the configured value, erase
  /// the first entry
  if (ctx->egl_frame_cache.size() > MAX_CACHED_EGL_FRAME_SIZE) {
    printf("egl_frame_cache size exceeds limit: %lu / %d\n", ctx->egl_frame_cache.size(),
           MAX_CACHED_EGL_FRAME_SIZE);
    // Erase first entry in the map
    // This is not a good strategy, as the first entry in the unorder_map may have just been
    // inserted.
    auto it = ctx->egl_frame_cache.begin();
    release_egl_frame_cache_entry(it);
    ctx->egl_frame_cache.erase(it);
  }
  auto item = std::make_tuple(resource, eglFrame, y_plane, uv_plane);
  return ctx->egl_frame_cache.insert(std::make_pair(surface_params, item));
}
#endif

// get y / uv plane address
static std::tuple<void *, void *> get_planes_addr(CustomCtx *ctx, NvBufSurface *surface,
                                                  int batch_index) {
  const NvBufSurfaceParams *surface_params = &surface->surfaceList[batch_index];
  if (surface_params->colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
      surface_params->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
    printf("unsupported colorformat %d \n", surface_params->colorFormat);
    return std::make_tuple(nullptr, nullptr);
  }

#if defined(__aarch64__)
  if(surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
    auto it = ctx->egl_frame_cache.find(surface_params);
    ///\ Get the cuda mem pointers by EGL mapping
    if (it == ctx->egl_frame_cache.end()) {
      auto ret = cache_egl_frame(ctx, surface, batch_index);
      if (ret.second) {
        it = ret.first;
      } else {
        printf("cache egl frame failed \n");
        return std::make_tuple(nullptr, nullptr);
      }
    }

    if (surface_params->layout == NVBUF_LAYOUT_PITCH) {
      void *luma = std::get<1>(it->second).frame.pPitch[0];
      void *chroma = std::get<1>(it->second).frame.pPitch[1];
      return std::make_tuple(luma, chroma);
    } else if (surface_params->layout == NVBUF_LAYOUT_BLOCK_LINEAR) {
      void *luma = reinterpret_cast<void *>(std::get<2>(it->second));
      void *chroma = reinterpret_cast<void *>(std::get<3>(it->second));
      return std::make_tuple(luma, chroma);
    } else {
      printf("unsupported layout %d \n", surface_params->layout);
      return std::make_tuple(nullptr, nullptr);
    }
  }
#endif
  auto input_height = surface->surfaceList[0].height;
  auto input_stride = surface->surfaceList[0].pitch;
  void *luma = surface->surfaceList[batch_index].dataPtr;
  void *chroma = static_cast<uint8_t *>(surface->surfaceList[batch_index].dataPtr) +
                 input_stride * input_height;
  return std::make_tuple(luma, chroma);
}

CustomCtx *initLib(CustomInitParams initparams) {
  auto ctx = std::make_unique<CustomCtx>();

  ctx->conv_handle = roiconv::create();
  auto it = initparams.user_configs.find(NVDSPREPROCESS_USER_CONFIGS_PIXEL_NORMALIZATION_FACTOR);
  if (it != initparams.user_configs.end()) {
    ctx->scales = split_string<3>(it->second);
  }

  it = initparams.user_configs.find(NVDSPREPROCESS_USER_CONFIGS_OFFSETS);
  if (it != initparams.user_configs.end()) {
    ctx->offsets = split_string<3>(it->second);
    printf("Using offsets: %f, %f, %f\n", ctx->offsets[0], ctx->offsets[1], ctx->offsets[2]);
    //convert "scale*(x-mean)"  to  "roi*alpha + beta"
    for(int i = 0; i < 3; i++){
      ctx->offsets[i] *= -ctx->scales[i];
    }
  }

  it = initparams.user_configs.find(NVDSPREPROCESS_USER_CONFIGS_SCALING_FILTER);
  if (it != initparams.user_configs.end()) {
    ctx->interpolation = to_roiconv_interpolation(it->second);
  }

  it = initparams.user_configs.find(NVDSPREPROCESS_USER_CONFIGS_SCALE_TYPE);
  if (it != initparams.user_configs.end()) {
    ctx->scale_type = static_cast<CustomCtx::ScaleType>(stoi(it->second));
    printf("%d is set for scale_type, ", ctx->scale_type);
    if(ctx->scale_type == CustomCtx::ScaleType::FitXY) {
      printf("using FitXY\n");
    } else if(ctx->scale_type == CustomCtx::ScaleType::FitCenter) {
      printf("using FitCenter\n");
    } else  if (ctx->scale_type == CustomCtx::ScaleType::Matrix) {
      printf("using Matrix\n");
      it = initparams.user_configs.find(NVDSPREPROCESS_USER_CONFIGS_AFFINE_MATRIX);
      if (it != initparams.user_configs.end()) {
        ctx->affine_matrix = split_string<6>(it->second);
      }
    } else {
      printf("unsupported scale_type, use default(FitXY)\n");
    }
  }

  printf("Using scales: %.5f, %.5f, %.5f\n", ctx->scales[0], ctx->scales[1], ctx->scales[2]);
  printf("affine_matrix: ");
  for (int i = 0; i < 6; i++) {
    printf("%.3f ", ctx->affine_matrix[i]);
  }
  printf("\n");

  // network initialization
  const auto &tensor_params = initparams.tensor_params;
  if (tensor_params.network_input_order == NvDsPreProcessNetworkInputOrder_kNCHW) {
    ctx->network_size.channels = tensor_params.network_input_shape[1];
    ctx->network_size.height = tensor_params.network_input_shape[2];
    ctx->network_size.width = tensor_params.network_input_shape[3];
  } else if (tensor_params.network_input_order == NvDsPreProcessNetworkInputOrder_kNHWC) {
    ctx->network_size.height = tensor_params.network_input_shape[1];
    ctx->network_size.width = tensor_params.network_input_shape[2];
    ctx->network_size.channels = tensor_params.network_input_shape[3];
  } else {
    printf("network-input-order = %d not supported\n", tensor_params.network_input_order);
    return nullptr;
  }
  switch (tensor_params.network_color_format) {
    case NvDsPreProcessFormat_RGB:
    case NvDsPreProcessFormat_BGR:
      if (ctx->network_size.channels != 3) {
        printf("RGB/BGR input format specified but network input channels is not 3\n");
        return nullptr;
      }
      break;
    case NvDsPreProcessFormat_GRAY:
      if (ctx->network_size.channels != 1) {
        printf("GRAY input format specified but network input channels is not 1.\n");
        return nullptr;
      }
      break;
    case NvDsPreProcessFormat_Tensor:
    default:
      printf("Unknown input format\n");
      return nullptr;
  }

  checkRuntime(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
  return ctx.release();
}

void deInitLib(CustomCtx *ctx) {
#if defined(__aarch64__)
  auto it = ctx->egl_frame_cache.begin();
  while (it != ctx->egl_frame_cache.end()) {
    release_egl_frame_cache_entry(it);
    ++it;
  }
  ctx->egl_frame_cache.clear();
#endif

  ctx->conv_handle.reset();
  checkRuntime(cudaStreamDestroy(ctx->stream));
  delete ctx;
}

NvDsPreProcessStatus CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                                          CustomTransformParams &params) {
  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                                               CustomTransformParams &params) {
  return NVDSPREPROCESS_SUCCESS;
}

NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer) {
  /** acquire a buffer from tensor pool */
  buf = acquirer->acquire();
  void *dst = buf->memory_ptr;
  GstBuffer *inbuf = (GstBuffer *)batch->inbuf;
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  if (!gst_buffer_map(inbuf, &inmap, GST_MAP_READ)) {
    GST_ERROR("input buffer mapinfo failed");
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }
  NvBufSurface *surface = (NvBufSurface *)inmap.data;
  gst_buffer_unmap(inbuf, &inmap);

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str = " tensorPrepare " + std::to_string(batch->inbuf_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxDomainRangePushEx(ctx->nvtx_domain, &eventAttrib);

  uint8_t fill_color[3] = {0, 0, 0};
  tensorParam.params.network_input_shape[0] = (int)batch->units.size();
  auto layer_bytes = tensorParam.params.buffer_size / tensorParam.params.network_input_shape[0];
  auto count = 0;
  for (const auto &it : batch->units) {
    roiconv::Task task;
    task.x0 = it.roi_meta.roi.left;
    task.y0 = it.roi_meta.roi.top;
    task.x1 = task.x0 + it.roi_meta.roi.width;
    task.y1 = task.y0 + it.roi_meta.roi.height;

    auto planes = get_planes_addr(ctx, surface, it.batch_index);
    task.input_planes[0] = std::get<0>(planes);
    task.input_planes[1] = std::get<1>(planes);
    task.input_planes[2] = nullptr;
    task.input_width = surface->surfaceList[0].width;
    task.input_height = surface->surfaceList[0].height;
    task.input_stride = surface->surfaceList[0].pitch;

    task.output_width = ctx->network_size.width;
    task.output_height = ctx->network_size.height;
    task.output = (uint8_t *)dst + layer_bytes * count++;
    if (ctx->scale_type == CustomCtx::ScaleType::FitXY) {
      task.resize_affine();
    } else if (ctx->scale_type == CustomCtx::ScaleType::FitCenter) {
      task.center_resize_affine();
    } else if (ctx->scale_type == CustomCtx::ScaleType::Matrix) {
      memcpy(task.affine_matrix, &ctx->affine_matrix[0], sizeof(ctx->affine_matrix));
    }
    memcpy(task.alpha, &ctx->scales[0], sizeof(ctx->scales));
    memcpy(task.beta, &ctx->offsets[0], sizeof(ctx->offsets));
    memcpy(task.fillcolor, fill_color, 3);
    ctx->conv_handle->add(task);
  }

  auto input_format = to_roiconv_input_format(&surface->surfaceList[0]);
  auto output_dtype = to_roiconv_output_dtype(tensorParam.params.data_type);
  auto output_format = to_roiconv_output_format(tensorParam.params.network_input_order,
                                                tensorParam.params.network_color_format);

  ctx->conv_handle->run(input_format, output_dtype, output_format, ctx->interpolation, ctx->stream,
                        false);
  checkRuntime(cudaStreamSynchronize(ctx->stream));
  nvtxDomainRangePop(ctx->nvtx_domain);

  return NVDSPREPROCESS_SUCCESS;
}
