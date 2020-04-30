################################################################################
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

APP:= deepstream-custom

CC:=g++

VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=5.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

DS_SRC_PATH?=
ifeq ($(DS_SRC_PATH),)
  $(error "DS_SRC_PATH is not set")
endif

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

PARSERS=nvdsinfer_customparser_dssd_tlt \
        nvdsinfer_customparser_frcnn_tlt \
	nvdsinfer_customparser_retinanet_tlt \
	nvdsinfer_customparser_ssd_tlt \
	nvdsinfer_customparser_yolov3_tlt

# Change to your deepstream SDK includes
CFLAGS+= -I$(DS_SRC_PATH)/sources/includes

SRCS:= $(wildcard *.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= $(SRCS:.c=.o)

CFLAGS+= `pkg-config --cflags $(PKGS)`

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(AT)$(foreach parser,$(PARSERS), make -C $(parser) &&) :
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	$(AT)$(foreach parser,$(PARSERS), make clean -C $(parser) &&) :
	rm -rf $(OBJS) $(APP)


