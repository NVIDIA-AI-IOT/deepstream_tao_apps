# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

CXX:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

# CUDA code generation flags
GPU_ARCHS	= 75 80 86 89 90 100 120
SMS = $(GPU_ARCHS)
ifeq ($(GENCODES),)
  $(foreach sm,$(SMS),$(eval GENCODES += -gencode arch=compute_$(sm),code=sm_$(sm)))
  GENCODES += -gencode arch=compute_90,code=compute_90
endif

ifeq ($(TARGET_DEVICE),aarch64)
    GENCODE_FLAGS := 
else
    GENCODE_FLAGS ?= $(GENCODES)
endif

KERNEL_PATH := ../modules/Lidar_AI_Solution/libraries/roiconvert/roi_conversion

SRCS:= nvdspreprocess_lib.cpp \
       $(KERNEL_PATH)/roi_conversion.cu

INCS:= $(wildcard *.h)
LIB:=libcustom2d_preprocess.so

CFLAGS += -fPIC -DHAVE_CONFIG_H -std=c++17 -Wall -Werror \
         -I/usr/local/cuda-$(CUDA_VER)/include \
         -I/opt/nvidia/deepstream/deepstream/sources/includes \
         -I/opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdspreprocess/include \
         -I$(KERNEL_PATH)

GST_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR ?= /opt/nvidia/deepstream/deepstream/lib/

LIBS := -shared -Wl,-no-undefined \
        -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
        -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
        -lnvbufsurface -lnvbufsurftransform -lcuda \
        -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(notdir $(OBJS:.cu=.o))

PKGS:= gstreamer-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS)
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: $(KERNEL_PATH)/%.cu $(KERNEL_PATH)/roi_conversion.hpp
	$(NVCC) -c -O3 -o $@ $(GENCODE_FLAGS) -I$(KERNEL_PATH) --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
