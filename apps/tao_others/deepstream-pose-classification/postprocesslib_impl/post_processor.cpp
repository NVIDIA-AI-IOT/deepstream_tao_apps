/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "post_processor.h"

using namespace std;

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the
 * DeepStreamSDK documentation.
 */
NvDsPostProcessStatus
ModelPostProcessor::parseLabelsFile(const std::string& labelsFilePath)
{
  std::ifstream labels_file(labelsFilePath);
  std::string delim{';'};
  if (!labels_file.is_open())
  {
    printError("Could not open labels file:%s", safeStr(labelsFilePath));
    return NVDSPOSTPROCESS_CONFIG_FAILED;
  }
  while (labels_file.good() && !labels_file.eof())
  {
    std::string line, word;
    std::vector<std::string> l;
    size_t pos = 0, oldpos = 0;

    std::getline(labels_file, line, '\n');
    if (line.empty())
      continue;

    while ((pos = line.find(delim, oldpos)) != std::string::npos)
    {
      word = line.substr(oldpos, pos - oldpos);
      l.push_back(word);
      oldpos = pos + delim.length();
    }
    l.push_back(line.substr(oldpos));
    m_Labels.push_back(l);
  }

  if (labels_file.bad())
  {
    printError("Failed to parse labels file:%s, iostate:%d",
        safeStr(labelsFilePath), (int)labels_file.rdstate());
    return NVDSPOSTPROCESS_CONFIG_FAILED;
  }
  return NVDSPOSTPROCESS_SUCCESS;
}


NvDsPostProcessStatus
ModelPostProcessor::initResource(NvDsPostProcessContextInitParams& initParams)
{

    if (!string_empty(initParams.labelsFilePath))
    {
      if (NVDSPOSTPROCESS_SUCCESS != parseLabelsFile(initParams.labelsFilePath)){
          printError("parse label file:%s failed", initParams.labelsFilePath);
      }
    }
    return NVDSPOSTPROCESS_SUCCESS;
}

