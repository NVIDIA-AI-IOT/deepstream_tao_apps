################################################################################
# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

APP:= deepstream-lpr-app

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= deepstream_lpr_app.cpp \
       deepstream_nvdsanalytics_meta.cpp \
       ../common/ds_yml_parse.cpp

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0
CFLAGS:=$(shell pkg-config --cflags $(PKGS))
CFLAGS+=-Wall -Werror -Wno-unused-function \
        -I/opt/nvidia/deepstream/deepstream/sources/includes \
        -I/usr/local/cuda/include \
        -I../common/

LIBS:= $(shell pkg-config --libs $(PKGS))
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lm -lstdc++\
       -lnvds_yml_parser -lyaml-cpp -lgstrtspserver-1.0 -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -L/usr/local/cuda/lib64/ -lcudart

all: $(APP)

%.o: %.cpp
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS)
	@$(CC) -o $(APP) $(OBJS) $(LIBS)
	@make -C nvinfer_custom_lpr_parser

clean:
	rm -rf $(OBJS) $(APP)
	@make -C nvinfer_custom_lpr_parser clean