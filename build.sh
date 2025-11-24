#!/bin/bash

# Build script for Tool Clustering System

set -e

echo "=== Building Tool Clustering System ==="

# Create build directory
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. || {
    echo "CMake configuration failed. Trying direct compilation..."
    cd ..
    g++ -std=c++17 -Wall -Wextra -O2 -pthread tool_clustering.cpp -o tool_clustering
    echo "Build complete! Executable: ./tool_clustering"
    exit 0
}

# Build
echo "Building..."
make -j$(nproc) || {
    echo "Make failed. Trying direct compilation..."
    cd ..
    g++ -std=c++17 -Wall -Wextra -O2 -pthread tool_clustering.cpp -o tool_clustering
    echo "Build complete! Executable: ./tool_clustering"
    exit 0
}

echo "Build successful! Executable: build/tool_clustering"
