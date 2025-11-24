# Simple Makefile for Tool Classifier
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I.
LDFLAGS = -pthread

# Target executable
TARGET = tool_classifier_standalone

# Source files
SRCS = tool_classifier_standalone.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET) tool_config.json

.PHONY: all clean run