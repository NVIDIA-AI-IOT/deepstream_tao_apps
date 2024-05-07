################################################################################
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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

APP:= deepstream-mdx-perception-app

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

SRCS:= deepstream_mdx_perception_app.c
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app.c 
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app_config_parser.c
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app_config_parser_yaml.cpp
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/src/*.c)
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/src/deepstream-yaml/*.cpp)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0

OBJS:= $(SRCS:.c=.o)
OBJS:= $(OBJS:.cpp=.o)

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes \
		 -I/opt/nvidia/deepstream/deepstream/sources/includes \
		 -I/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app \
		 -I/usr/local/cuda/include
#		 -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \

LIBS:= -L/usr/local/cuda/lib64/ -lcudart
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper \
	   -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm \
       -lyaml-cpp -lcuda -lgstrtspserver-1.0 -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS+= $(shell pkg-config --libs $(PKGS))

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)

