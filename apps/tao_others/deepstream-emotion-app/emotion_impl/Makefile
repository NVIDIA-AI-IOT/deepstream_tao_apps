################################################################################
# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
#################################################################################

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CXX:= g++

SRCS:= emotion_impl.cpp ../../common/ds_yml_parse.cpp

INCS:= $(wildcard *.h)
LIB:=libnvds_emotion_impl.so

CFLAGS+= -I /usr/local/cuda/include \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes -I../includes \
	 -I /opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers \
	 -I /opt/nvidia/deepstream/deepstream/sources/gst-plugins/gst-nvdsvideotemplate/includes \
	 -I ../../deepstream-faciallandmark-app \
	 -I ../ -I ../../common/

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

LIBS := -shared -Wl,-no-undefined -lnvds_yml_parser -lgstrtspserver-1.0 \
	-L/usr/local/cuda/lib64/ -lcudart -ldl -lpthread -lyaml-cpp \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
	-lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool -lnvds_infer\
	-L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs -lnvcv_emotions -lnvcv_core -lnvcv_trtbackend \
	-Wl,-rpath,$(LIB_INSTALL_DIR) \
	-Wl,-rpath,$(LIB_INSTALL_DIR)/cvcore_libs/

ifeq ($(TARGET_DEVICE),aarch64)
  LIBS += -L/usr/lib/aarch64-linux-gnu/tegra
endif


OBJS:= $(SRCS:.cpp=.o)
PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))
CFLAGS += -Wno-deprecated-declarations -fPIC

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

$(DEP): $(DEP_FILES)
	$(MAKE) -C customlib_impl/

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
