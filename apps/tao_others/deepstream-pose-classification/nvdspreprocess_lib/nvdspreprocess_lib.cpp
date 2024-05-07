/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <string.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <condition_variable>
#include <sys/time.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "nvdspreprocess_lib.h"
#include "nvdsmeta_schema.h"
#include <mutex>
#include <vector>

using std::vector;

#define    _MIN_FRAME_                   3
#define    _MAX_FRAME_                   300
#define    _MAX_OBJECT_NUM_              20
#define    _TIME_OUT_                    2
#define FREE(p) (free(p), p=NULL)

/*wrap keypoints*/
struct   SObjectContex
{
  uint64_t object_id;
  float* x, *y, *z;
  int frameIndex;
  bool firstUse;
  long  tv_sec;
  SObjectContex() {
    object_id = UNTRACKED_OBJECT_ID;
    x = y = z = NULL;
    frameIndex = 0;
    firstUse = true;
    tv_sec = 0;
  };
  ~SObjectContex() {
    FREE(x);
    FREE(y);
    FREE(z);
  };
};

struct CustomCtx
{
  std::mutex mtx;
  /* vector for obejct context*/
  vector<SObjectContex*> multi_objects;
  int one_channel_element_num;
  int two_channel_element_num;
  int move_element_num;
  ~CustomCtx()
  {
    int size = multi_objects.size();
    printf("size:%d\n", size);
    for (int i = 0; i < size; i++)
    {
      delete multi_objects[i];
      multi_objects[i] = NULL;
    }
    multi_objects.clear();
  };
};

/* find Object by object_id */
SObjectContex* findObjectCtx(CustomCtx *ctx, guint64 object_id)
{
  std::unique_lock <std::mutex> lck(ctx->mtx);
  SObjectContex* pSObjectCtx = NULL;
  for (vector<SObjectContex*>::iterator itor = ctx->multi_objects.begin();
    itor != ctx->multi_objects.end(); itor++) {
    if( (*itor)->object_id == object_id) {
      pSObjectCtx = (*itor);
    }
  }
  return pSObjectCtx;
}

/* find unused object by object_id */
SObjectContex*
findUnusedObjectCtx(CustomCtx *ctx, guint64 object_id)
{
  std::unique_lock <std::mutex> lck(ctx->mtx);
  SObjectContex* pSObjectCtx = NULL;
  for (vector<SObjectContex*>::iterator itor = ctx->multi_objects.begin();
    itor != ctx->multi_objects.end(); itor++) {
    if( (*itor)->object_id == UNTRACKED_OBJECT_ID) {
      pSObjectCtx = (*itor);
    }
  }
  return pSObjectCtx;
}

/* extend object group */
SObjectContex* 
CreateObjectCtx(CustomCtx *ctx)
{
  std::unique_lock <std::mutex> lck(ctx->mtx);
  SObjectContex* pSObjectCtx = new SObjectContex;
  if(pSObjectCtx)
  {
    pSObjectCtx->object_id = UNTRACKED_OBJECT_ID;
    pSObjectCtx->x = (float*)calloc(ctx->one_channel_element_num, sizeof(float));
    pSObjectCtx->y = (float*)calloc(ctx->one_channel_element_num, sizeof(float));
    pSObjectCtx->z = (float*)calloc(ctx->one_channel_element_num, sizeof(float));
    pSObjectCtx->frameIndex = 0;
    ctx->multi_objects.push_back(pSObjectCtx);
  }
  return pSObjectCtx;
}

/* extend object group */
void
ResetObjectCtx(CustomCtx *ctx, SObjectContex *pSObjectCtx)
{
  if(pSObjectCtx)
  {
    printf("ResetObjectCtx, object_id:%ld\n", pSObjectCtx->object_id);
    pSObjectCtx->object_id = UNTRACKED_OBJECT_ID;
    memset(pSObjectCtx->x, 0, ctx->one_channel_element_num * sizeof(float));
    memset(pSObjectCtx->y, 0, ctx->one_channel_element_num * sizeof(float));
    memset(pSObjectCtx->z, 0, ctx->one_channel_element_num * sizeof(float));
    pSObjectCtx->frameIndex = 0;
    pSObjectCtx->tv_sec = 0;
  }
}

void LoopObjectCtx(CustomCtx *ctx)
{
  std::unique_lock <std::mutex> lck(ctx->mtx);
  struct timeval tv;
  SObjectContex* pSObjectCtx = NULL;
  for (vector<SObjectContex*>::iterator itor = ctx->multi_objects.begin();
  itor != ctx->multi_objects.end(); itor++) {
    pSObjectCtx = (*itor);
    gettimeofday (&tv, NULL);
    if(pSObjectCtx->object_id != UNTRACKED_OBJECT_ID &&
      (tv.tv_sec - pSObjectCtx->tv_sec) > _TIME_OUT_){
      ResetObjectCtx(ctx, pSObjectCtx);
    }
  }
}

/* save 34 keypoints to local */
void
sveKeypoints(CustomCtx *ctx, void *user_meta_data, SObjectContex *pSObjectCtx)
{
  std::unique_lock <std::mutex> lck(ctx->mtx);
  if(pSObjectCtx)
  {
    NvDsJoints *ds_joints = (NvDsJoints *) user_meta_data;
    //move from tail to head
    memmove(pSObjectCtx->x, pSObjectCtx->x + 34, ctx->move_element_num * sizeof(float));
    memmove(pSObjectCtx->y, pSObjectCtx->y + 34, ctx->move_element_num * sizeof(float));
    memmove(pSObjectCtx->z, pSObjectCtx->z + 34, ctx->move_element_num * sizeof(float));

    //save keypoints
    for(int i = 0; i < ds_joints->num_joints; i++){
        *(pSObjectCtx->x + ctx->move_element_num + i) = ds_joints->joints[i].x;
        *(pSObjectCtx->y + ctx->move_element_num + i)  = ds_joints->joints[i].y;
        *(pSObjectCtx->z + ctx->move_element_num + i)  = ds_joints->joints[i].z;
    }

    //update time
    struct timeval tv;
    gettimeofday (&tv, NULL);
    pSObjectCtx->tv_sec = tv.tv_sec;
  }
}

NvDsPreProcessStatus
CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch, NvDsPreProcessCustomBuf *&buf,
                        CustomTensorParams &tensorParam, NvDsPreProcessAcquirer *acquirer)
{
  NvDsPreProcessStatus status = NVDSPREPROCESS_TENSOR_NOT_READY;

  /** acquire a buffer from tensor pool */
  buf = acquirer->acquire();
  float * pDst = (float*)buf->memory_ptr;
  int units = batch->units.size();
  for(int i = 0; i < units; i++)
  {
    guint64 object_id = batch->units[i].roi_meta.object_meta->object_id;
    GstBuffer *inbuf = (GstBuffer *)batch->inbuf;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsMetaList *l_user = NULL;
    SObjectContex* pSObjectCtx = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next)
    {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
      for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next)
      {
        NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
        if(obj_meta->object_id != object_id)
          continue;

        for (l_user = obj_meta->obj_user_meta_list; l_user != NULL;
              l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
          if (user_meta->base_meta.meta_type == NVDS_OBJ_META)
          {
            /* find by objectid */
            pSObjectCtx = findObjectCtx(ctx, obj_meta->object_id);
            if(!pSObjectCtx)
            {
              /* can't find objectid, find one whose objectid is -1 */
              pSObjectCtx = findUnusedObjectCtx(ctx, obj_meta->object_id);
              if(pSObjectCtx)
              {
                /*can find one whose objectid is not -1, copy keypoints*/
                pSObjectCtx->object_id = obj_meta->object_id;
                sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
              } else {
                /* if no, extent Skeypoints, then copy keypoints*/
                pSObjectCtx = CreateObjectCtx(ctx);
                printf("extendObjectCtx pSObjectCtx:%p\n", pSObjectCtx);
                if(pSObjectCtx)
                {
                  pSObjectCtx->object_id = obj_meta->object_id;
                  sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
                }
              }
            } else {
              /* can find, copy keypoints */
              sveKeypoints(ctx, user_meta->user_meta_data, pSObjectCtx);
            }
          }
        }
      }
    }

    /* copy to buffer, 3 X 300 X 34 X 1 (C T V M) */
    if(pSObjectCtx)
    {
      int bufLen = ctx->one_channel_element_num*sizeof(float);
      cudaMemcpy(pDst, pSObjectCtx->x, bufLen, cudaMemcpyHostToDevice);
      cudaMemcpy(pDst + ctx->one_channel_element_num, pSObjectCtx->y, bufLen, cudaMemcpyHostToDevice);
      cudaMemcpy(pDst + ctx->two_channel_element_num, pSObjectCtx->z, bufLen, cudaMemcpyHostToDevice);
      pDst = pDst + 3*bufLen;
    }
  }

  //reset object context if timeout
  LoopObjectCtx(ctx);
  status = NVDSPREPROCESS_SUCCESS;
  return status;
}

NvDsPreProcessStatus
CustomTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf, CustomTransformParams &params)
{
  /* do nothing, bodypose data is in object's metadata, here we can't access object */
  return NVDSPREPROCESS_SUCCESS;
}

CustomCtx *initLib(CustomInitParams initparams)
{
  CustomCtx *ctx =  new CustomCtx;
  std::string sframeSeqLen = initparams.user_configs[NVDSPREPROCESS_USER_CONFIGS_FRAMES_SEQUENCE_LENGHTH];
  int len = atoi(sframeSeqLen.c_str());
  printf("frameSeqLen:%d\n", len);
  if(len < _MIN_FRAME_ || len > _MAX_FRAME_) 
  {
    printf("frameSeqLen iilegal, use default vaule 300\n");
    len = _MAX_FRAME_;
  }
  ctx->one_channel_element_num = len * 34;
  ctx->two_channel_element_num = 2 * len * 34;
  ctx->move_element_num = (len-1) * 34;

  /* initial vector for multi_keypoints*/
  for(int i = 0; i < _MAX_OBJECT_NUM_; i++){
    CreateObjectCtx(ctx);
  }

  return ctx;
}

void deInitLib(CustomCtx *ctx)
{
  delete ctx;
}
