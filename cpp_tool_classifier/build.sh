#!/bin/bash
# Build script for Tool Classifier

set -e  # Exit on error

echo "=== Building Tool Classifier ==="
echo

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Run CMake with g++ compiler
echo "Running CMake..."
CXX=g++ cmake ..

# Build
echo "Building..."
make -j$(nproc)

echo
echo "=== Build Complete ==="
echo "Executable: ./build/tool_classifier"
echo
echo "To run:"
echo "  cd build"
echo "  ./tool_classifier --help"
echo
