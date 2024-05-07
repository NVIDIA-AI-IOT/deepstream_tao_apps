################################################################################
# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
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

APP:= deepstream-gesture-app

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

SRCS:= $(wildcard *.c)
CPPSRCS+= $(wildcard *.cpp)
CPPSRCS+= ../deepstream-bodypose2d-app/deepstream_bodypose2d_meta.cpp \
          ../common/ds_yml_parse.cpp

INCS:= $(wildcard *.h)
INCS+= ../deepstream-bodypose2d-app/ds_bodypose2d_meta.h \
       ../common/ds_yml_parse.h

PKGS:= gstreamer-1.0

COBJS:= $(SRCS:.c=.o)
CPPOBJS:= $(CPPSRCS:.cpp=.o)
OBJS:= $(COBJS) $(CPPOBJS)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
	 -I/opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers \
         -I /usr/local/cuda/include \
         -I ../deepstream-bodypose2d-app -I ../common

CFLAGS+= `pkg-config --cflags $(PKGS)`

CFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-sign-compare -Wno-deprecated-declarations

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_inferutils \
       -lnvds_utils -lm -lstdc++ -lnvds_yml_parser -lgstrtspserver-1.0 \
       -L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs \
       -L/usr/local/cuda/lib64/ -lcudart -lcuda -lyaml-cpp \
       -lnvcv_bodypose2d -lnvcv_core -lnvcv_tensorops -lnvcv_trtbackend \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o : %.c $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o : %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)


