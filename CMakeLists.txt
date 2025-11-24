cmake_minimum_required(VERSION 3.14)
project(ToolClassifier VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find packages
find_package(Threads REQUIRED)

# FetchContent for downloading dependencies
include(FetchContent)

# Download and configure nlohmann/json
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# ONNX Runtime
# Option 1: Use pre-installed ONNX Runtime
# find_package(onnxruntime REQUIRED)

# Option 2: Download ONNX Runtime
set(ONNXRUNTIME_VERSION "1.16.3")
if(WIN32)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
elseif(APPLE)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
else()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
endif()

FetchContent_Declare(
    onnxruntime
    URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(onnxruntime)

# Set ONNX Runtime paths
set(ONNXRUNTIME_ROOT_PATH ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT_PATH}/include)
if(WIN32)
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_PATH}/lib)
else()
    set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_ROOT_PATH}/lib)
endif()

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)

# Main executable
add_executable(tool_classifier tool_classifier.cpp)

# Include directories
target_include_directories(tool_classifier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(tool_classifier PRIVATE
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

# Copy ONNX Runtime shared libraries to build directory
if(WIN32)
    add_custom_command(TARGET tool_classifier POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll
        $<TARGET_FILE_DIR:tool_classifier>
    )
endif()

# Install targets
install(TARGETS tool_classifier
    RUNTIME DESTINATION bin
)

# Install configuration file
install(FILES tool_config.json
    DESTINATION bin
)

# Create a simple version without ONNX dependency for testing
option(BUILD_SIMPLE_VERSION "Build simplified version without ONNX Runtime" OFF)

if(BUILD_SIMPLE_VERSION)
    add_executable(tool_classifier_simple tool_classifier_simple.cpp)
    target_link_libraries(tool_classifier_simple PRIVATE
        nlohmann_json::nlohmann_json
        Threads::Threads
    )
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")