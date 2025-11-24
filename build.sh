#!/bin/bash

# Tool Classifier Build Script
# This script builds the C++ tool classification and clustering system

set -e

echo "========================================"
echo "Building Tool Classification System"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_requirement() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        echo "Please install $1 and try again"
        exit 1
    fi
}

echo -e "\n${YELLOW}Checking requirements...${NC}"
check_requirement cmake
check_requirement g++
check_requirement git

# Parse command line arguments
BUILD_TYPE="Release"
BUILD_SIMPLE="OFF"
CLEAN_BUILD="OFF"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --simple)
            BUILD_SIMPLE="ON"
            shift
            ;;
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug    Build in debug mode"
            echo "  --simple   Build simplified version (without ONNX Runtime)"
            echo "  --clean    Clean build directory before building"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create build directory
BUILD_DIR="build"
if [ "$CLEAN_BUILD" == "ON" ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf $BUILD_DIR
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo -e "\n${YELLOW}Configuring with CMake...${NC}"
echo "Build type: $BUILD_TYPE"
echo "Build simple version: $BUILD_SIMPLE"

cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DBUILD_SIMPLE_VERSION=$BUILD_SIMPLE \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
echo -e "\n${YELLOW}Building...${NC}"
cmake --build . -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Build successful!${NC}"
    
    # List built executables
    echo -e "\n${GREEN}Built executables:${NC}"
    if [ -f "tool_classifier_simple" ]; then
        echo "  - tool_classifier_simple (simplified version)"
    fi
    if [ -f "tool_classifier" ]; then
        echo "  - tool_classifier (full version with ONNX Runtime)"
    fi
    
    # Copy config file to build directory
    cp ../tool_config.json . 2>/dev/null || true
    
    echo -e "\n${GREEN}To run the tool classifier:${NC}"
    echo "  cd build"
    echo "  ./tool_classifier_simple [config_file]"
    echo ""
    echo "Or for the full version (if built):"
    echo "  ./tool_classifier [config_file]"
else
    echo -e "\n${RED}Build failed!${NC}"
    exit 1
fi