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

#include "debug_logger_raii.hpp"
#include <iostream>
#include <ctime>


const bool DebugConfig::enabled = []() {
    const char* debug_env = getenv(ENABLE_ENV_NAME);
    if (!debug_env) {
        std::cout << "[" << __FILE__ << "] "
                  << "Environment variable " << ENABLE_ENV_NAME << " not set, DEBUG_DUMP disabled" 
                  << std::endl;
        return false;
    }
    bool is_enabled = (strcmp(debug_env, "1") == 0 || 
                      strcmp(debug_env, "true") == 0 || 
                      strcmp(debug_env, "TRUE") == 0);
    std::cout << "[" << __FILE__ << "] "
              << "Environment variable " << ENABLE_ENV_NAME << "=" << debug_env 
              << ", DEBUG_DUMP " << (is_enabled ? "enabled" : "disabled") 
              << std::endl;
    return is_enabled;
}();

std::string DebugLoggerRAII::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = std::localtime(&now_c);
    
    std::stringstream ss;
    ss << std::put_time(now_tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string DebugLoggerRAII::GetLogHeader() {
    return "[" + GetTimestamp() + "] [" + func_name + "] ";
}

DebugLoggerRAII::DebugLoggerRAII(const char* func, int line, bool is_enabled) 
    : func_name(func), start_line(line), enabled(is_enabled) {
    if (!enabled) return;

    filename = std::string("/tmp/debug_") + func_name + "_" + 
                          std::to_string(std::time(nullptr)) + ".log";
    log_file.open(filename);
    if (log_file.is_open()) {
        std::cout << GetLogHeader() << "DEBUG_DUMP started at file: " << filename << std::endl;
        log_file << GetLogHeader() << "DEBUG_DUMP started at line " << line << std::endl;
    } else {
        std::cerr << GetLogHeader() << "Failed to create debug log file: " << filename << std::endl;
    }
}

DebugLoggerRAII::~DebugLoggerRAII() {
    if (enabled && log_file.is_open()) {
        log_file << GetLogHeader() << "DEBUG_DUMP ended" << std::endl;
        log_file.close();
        std::cout << GetLogHeader() << "DEBUG_DUMP closed log file: " << filename << std::endl;
    }
}

void DebugLoggerRAII::log(const char* format, ...) {
    if (!enabled || !log_file.is_open()) return;
    
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    log_file << GetLogHeader() << buffer << std::endl;
}
