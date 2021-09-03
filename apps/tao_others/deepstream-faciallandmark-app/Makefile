################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

APP:= deepstream-faciallandmark-app

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= deepstream_faciallandmark_app.o deepstream_faciallandmark_meta.o

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
	 -I/opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers \
         -I /usr/local/cuda-$(CUDA_VER)/include

CFLAGS+= `pkg-config --cflags $(PKGS)`

CFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-sign-compare

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_inferutils \
       -lnvds_utils -lm -lstdc++ \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda \
       -L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs \
       -lnvcv_faciallandmarks -lnvcv_core -lnvcv_tensorops -lnvcv_trtbackend \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

deepstream_faciallandmark_app.o: deepstream_faciallandmark_app.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -fpermissive -Wall -Werror $(CFLAGS) $<

deepstream_faciallandmark_meta.o: deepstream_faciallandmark_meta.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall -Werror $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)


