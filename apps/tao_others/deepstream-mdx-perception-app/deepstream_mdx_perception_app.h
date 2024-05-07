/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __DEEPSTREAM_FEWSHOT_LEARNING_APP_H__
#define __DEEPSTREAM_FEWSHOT_LEARNING_APP_H__

#include <gst/gst.h>
#include "deepstream_config.h"

/* From deepstream_utc.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE

typedef struct
{
  gint anomaly_count;
  gint meta_number;
  struct timespec timespec_first_frame;
  GstClockTime gst_ts_first_frame;
  GMutex lock_stream_rtcp_sr;
  guint32 id;
  gint frameCount;
  GstClockTime last_ntp_time;
} StreamSourceInfo;

typedef struct
{
  StreamSourceInfo streams[MAX_SOURCE_BINS];
} TestAppCtx;

/** URI sample:
 * HWY_20_AND_BRYANT__WB__4_11_2018_4_59_59_485_AM_UTC-07_00.mp4
 *
 * Specification format:
 * __M_DD_YYYY_H_MIN_SEC_MSEC_AM/PM_UTC<Offset>.mp4
 * a) Offset is:
 * <+/-HOURS_MIN>
 * b) In ..AM/PM_TIMEZONE<Offset>.mp4
 * TIMEZONE will always be UTC
 *
 * The __M_DD_YYYY_H_MIN_SEC_MSEC_AM/PM time
 * shall <Offset-period> behind UTC
 *
 */
#define URI_UTC_START_DELIM "__"
#define URI_UTC_END_DELIM "_UTC"
#define MAX_UTC_STRING_LEN (256)
#define LENGTH__AMPM_UTC (3)  //"_AM"
#define LENGTH_UTC_END_DELIM (4)

static gboolean extract_ms_from_utc(gchar *utc, guint32 *ms) {
  /** find _UTC delim */
  gchar *utc_delim = (gchar *)strstr(utc, URI_UTC_END_DELIM);
  gint32 utc_string_length;
  if (!utc_delim) {
    return FALSE;
  }

  /** find the immediately preceeding '_' */
  utc_string_length =
      (guint32)(((size_t)utc_delim) - ((size_t)utc)) - LENGTH__AMPM_UTC - 1;
  gint32 i_ms = 0; /**< index into utc at which ms is */
  for (i_ms = utc_string_length; ((i_ms >= 0) && (utc[i_ms] != '_')); i_ms--)
    ;
  if (i_ms < 0) {
    return FALSE;
  }

  guint32 input_items = sscanf(&utc[i_ms], "_%u_", ms);
  if (input_items != 1) {
    return FALSE;
  }

  /** also remove the ms part from utc string */
  strncpy(&utc[i_ms], (utc_delim - LENGTH__AMPM_UTC),
          MAX_UTC_STRING_LEN - utc_string_length);

  return TRUE;
}

/**
 * @brief  Extracts and returns the <Offset> in nanoseconds
 */
static gboolean extract_offset_from_utc(gchar *utc, gint64 *offset_nsec) {
  /** find _UTC delim */
  gchar *utc_offset = (gchar *)strstr(utc, URI_UTC_END_DELIM);
  if (!utc_offset) {
    return FALSE;
  }
  gint32 hours;
  guint32 minutes;
  utc_offset += LENGTH_UTC_END_DELIM;
  sscanf(utc_offset, "%d_%u", &hours, &minutes);
  *offset_nsec = ((ABS(hours) * 60 + minutes) * 60) * GST_SECOND;
  if (hours > 0) {
    /** if positive, offset shall be subtracted from
     * __M_DD_YYYY_H_MIN_SEC_MSEC_AM/PM to arrive at UTC
     * Otherwise added
     */
    *offset_nsec = (*offset_nsec) * -1;
  }
  return TRUE;
}

struct timespec extract_utc_from_uri(gchar *uri) {
  gchar utc_string[MAX_UTC_STRING_LEN];
  struct tm utc_tmbroken = {0};
  struct timespec utc_timespec = {0};
  gchar *utc_iter = uri;
  gchar *utc = NULL;
  guint32 ms = 0;
  gint64 time_ns = 0;
  /** Find the beginning of UTC time field in URI
   * The URI delimiter
   */
  do {
    utc = utc_iter;
    utc_iter = (gchar *)strstr(utc, URI_UTC_START_DELIM);
    if (utc_iter) {
      /** skip the starting delim */
      utc_iter += strlen(URI_UTC_START_DELIM);
    }
  } while (utc_iter);

  if (utc == uri) {
    /** Invalid URI */
    return utc_timespec;
  }

  /** extract ms and remove ms_ */
  g_strlcpy(utc_string, utc, MAX_UTC_STRING_LEN);

  gboolean ok = extract_ms_from_utc(utc_string, &ms);
  if (!ok) {
    /** Invalid URI */
    return utc_timespec;
  }

  /** First generate the time tm structure from provided string;
   * Note: Assuming UTC always */
  gchar *first_char_not_processed =
      strptime((const char *)utc_string, "%m_%d_%Y_%I_%M_%S_%p", &utc_tmbroken);
  if (!first_char_not_processed ||
      strncmp(first_char_not_processed, URI_UTC_END_DELIM,
              LENGTH_UTC_END_DELIM) != 0) {
    /** first_char_not_processed should be URI_UTC_END_DELIM
     * Otherwise, it is an error condition */
    return utc_timespec;
  }

  ok = extract_offset_from_utc(utc_string, &time_ns);
  if (!ok) {
    /** Invalid URI */
    return utc_timespec;
  }

  /** mktime(): Now convert the broken down tm struct to time_t, calendar time
   * representation*/
  utc_timespec.tv_sec = timegm(&utc_tmbroken);
  utc_timespec.tv_nsec = ms * GST_MSECOND;

  /** final time is */
  time_ns = (gint64)(GST_TIMESPEC_TO_TIME(utc_timespec)) + time_ns;

  GST_TIME_TO_TIMESPEC(time_ns, utc_timespec);

  return utc_timespec;
}

#endif /**< __DEEPSTREAM_TEST5_APP_H__ */
