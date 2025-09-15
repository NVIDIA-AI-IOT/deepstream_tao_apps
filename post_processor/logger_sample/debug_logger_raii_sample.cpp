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

#include "../debug_logger_raii.hpp"
#include <vector>

// Sample function demonstrating debug logger usage
void processData(const std::vector<float>& data) {
    // Start debug section
    {
        DEBUG_DUMP_SECTION();
        
        DEBUG_DUMP("Processing %zu elements", data.size());
        
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i] > 0.5f) {
                DEBUG_DUMP("Element %zu: %f exceeds threshold", i, data[i]);
            }
        }
        
        DEBUG_DUMP("Processing complete");
    } // Debug logger automatically closes here
}

// Sample main function for testing
int main() {
    std::vector<float> test_data = {0.1f, 0.6f, 0.3f, 0.8f, 0.2f};
    // expose DEBUG=1
    // unset DEBUG
    processData(test_data);
    return 0;
}
