CC := g++
CFLAGS := -Wall -std=c++11 -I..

# Debug flags
CFLAGS_DEBUG := $(CFLAGS) -g -DDEBUG

# Source files
SRCFILES := debug_logger_raii_sample.cpp
SRCFILES += ../debug_logger_raii.cpp
TARGET := debug_logger_raii_sample

# Default target
all: $(TARGET)

$(TARGET): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS_DEBUG)

clean:
	rm -rf $(TARGET)

.PHONY: all clean
