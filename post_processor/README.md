# DeepStream Post-Processor 

## Build

### Release build
```bash
CUDA_VER=12.6 make
```
libnvds_infercustomparser_tao.so will be generated in current directory

### Debug build
```bash
CUDA_VER=12.6 make debug
```
libnvds_infercustomparser_tao_debug.so will be generated in current directory

Both builds have below debug utilities, that can be enabled/disabled by setting DEBUG environment variable. The difference is debug build has gdb symbol.

## Debug Utilities
### Overview
This directory contains debug utilities for DeepStream post-processing:
- RAII-based file logger for detailed debugging
- Tensor-specific debug utilities for visualizing tensor data
- Zero-overhead when disabled (single environment variable check)
- Thread-safe implementation
- These src files are self independent, can be easily integrated into any Deepstream lib

### Debug Macros

#### 1. DEBUG_DUMP_SECTION and DEBUG_DUMP
RAII-based file logger for detailed debugging:
```cpp
{
    DEBUG_DUMP_SECTION(); // Creates a scoped logger
    DEBUG_DUMP("Processing %zu elements", data.size());
    for (size_t i = 0; i < data.size(); i++) {
        DEBUG_DUMP("Element %zu: %f", i, data[i]);
    }
} // Logger automatically closes file
```
- Creates timestamped log file in /tmp
- Zero overhead when disabled
- Automatically closes file when scope ends
- Thread-safe (each instance has its own file)

#### 2. DEBUG_DS_TENSOR
DeepStream tensor visualization:
```cpp
DEBUG_DS_TENSOR(layer_info); // Prints truncated tensor preview
```
- Prints dimensions and first few elements
- Full tensor data dumped to file
- Supports FLOAT, INT32, INT64 types
- Shows tensor shape and memory layout

#### 3. DEBUG_TENSOR
Generic tensor debug utility:
```cpp
DEBUG_TENSOR("input", data.data(), data.size(), float);
```
- Prints first 100 elements
- Full data dumped to file
- Type-safe implementation

#### 4. DEBUG_PRINT
Simple debug print utility:
```cpp
DEBUG_PRINT("Processing data: %d", value);
```
- Lightweight console output
- Includes timestamp
- No file I/O

### Usage

#### Enable/Disable Debug

```bash
# Enable all debug output
export DEBUG=1

# Disable all debug output
unset DEBUG
```

#### Performance Impact
- Disabled: Zero overhead (single static bool check)
- Enabled: File I/O only occurs within DEBUG_DUMP_SECTION scope
- Thread-safe with minimal locking
- Static initialization occurs once at program start