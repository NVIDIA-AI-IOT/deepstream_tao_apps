/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DEBUG_LOGGER_RAII_H
#define DEBUG_LOGGER_RAII_H

#include <fstream>
#include <string>
#include <cstdarg>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iostream>

#define ENABLE_ENV_NAME "DEBUG"

/**
 * @brief Debug logger class using RAII (Resource Acquisition Is Initialization)
 * to automatically manage the log file.
 * 
 * This class is used to log debug messages to a file.
 * The log file is automatically closed when the object is destroyed.
 * With the help of the DEBUG_DUMP_SECTION macro, we can start and end a debug section.
 * Advantages of using this class:
 * - Automatically closes the log file even if there's an early return or exception.
 * - clearly defines the scope of debug logging using {} block
 * - No need to remember to close anything
 * - Thread-safe as each instance has its own file handle
 * - Object construction and destruction has minimal overhead when it is disabled
 * 
 * @param func The function name
 * @param line The line number
 * @param is_enabled Whether the logger is enabled
 * 
 * @note The log file is created in the /tmp directory.
 */
class DebugLoggerRAII {
public:
    DebugLoggerRAII(const char* func, int line, bool is_enabled);
    ~DebugLoggerRAII();
    void log(const char* format, ...);

private:
    std::string filename;
    std::ofstream log_file;
    const char* func_name;
    int start_line;
    bool enabled;
    
    // Helper function to get formatted timestamp
    static std::string GetTimestamp();
    // Helper function to get log header with timestamp and file info
    std::string GetLogHeader();
};


class DebugConfig {
private:
    static const bool enabled;
public:
    static bool IsEnabled() { return enabled; }
};

// Debug macros
#define DEBUG_DUMP_SECTION() \
    DebugLoggerRAII debug_logger(__func__, __LINE__, DebugConfig::IsEnabled())

#define DEBUG_DUMP(...) \
    debug_logger.log(__VA_ARGS__)

#endif // DEBUG_LOGGER_RAII_H