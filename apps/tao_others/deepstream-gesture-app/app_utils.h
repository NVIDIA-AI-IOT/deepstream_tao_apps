/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <gst/gst.h>
#include <glib.h>

#define LOG_FORMAT_(fmt) "%s:%d " fmt, __FILE__, __LINE__
static const char* log_enable = std::getenv("ENABLE_DEBUG");

#define AppLogE(fmt, ...)                                           \
    do {                                                                 \
        g_printerr(                                               \
            LOG_FORMAT_(fmt), ##__VA_ARGS__); \
    } while (0)

#define AppLogD(fmt, ...)                                           \
    do {                                                                 \
        if(log_enable && std::stoi(log_enable)) {               \
            g_print(                                               \
                LOG_FORMAT_(fmt), ##__VA_ARGS__); \
        }                                               \
    } while (0)
