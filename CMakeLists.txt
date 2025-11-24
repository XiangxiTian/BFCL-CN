cmake_minimum_required(VERSION 3.15)
project(ToolClustering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Optional: Use ONNX Runtime for actual Gemma model inference
option(USE_ONNX "Use ONNX Runtime for model inference" OFF)

if(USE_ONNX)
    # Find ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
            /usr/local/include/onnxruntime
            /opt/onnxruntime/include
    )
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib
            /usr/local/lib
            /opt/onnxruntime/lib
    )
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
        add_definitions(-DUSE_ONNX)
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
    else()
        message(WARNING "ONNX Runtime not found. Using mock implementation.")
        set(USE_ONNX OFF)
    endif()
endif()

# Optional: Use nlohmann/json for better JSON parsing
option(USE_NLOHMANN_JSON "Use nlohmann/json library" OFF)

if(USE_NLOHMANN_JSON)
    # Try to find nlohmann/json (header-only library)
    find_path(NLOHMANN_JSON_INCLUDE_DIR
        NAMES nlohmann/json.hpp
        PATHS
            ${CMAKE_SOURCE_DIR}/third_party/json/include
            /usr/local/include
            /usr/include
    )
    
    if(NLOHMANN_JSON_INCLUDE_DIR)
        include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
        add_definitions(-DUSE_NLOHMANN_JSON)
        message(STATUS "nlohmann/json found: ${NLOHMANN_JSON_INCLUDE_DIR}")
    else()
        message(WARNING "nlohmann/json not found. Using simple regex parser.")
        set(USE_NLOHMANN_JSON OFF)
    endif()
endif()

# Source files
set(SOURCES
    tool_clustering.cpp
)

# Create executable
add_executable(tool_clustering ${SOURCES})

# Link libraries
if(USE_ONNX AND ONNXRUNTIME_LIBRARY)
    target_link_libraries(tool_clustering ${ONNXRUNTIME_LIBRARY})
endif()

# Threading library (for future parallel processing)
find_package(Threads REQUIRED)
target_link_libraries(tool_clustering Threads::Threads)

# Installation
install(TARGETS tool_clustering
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "USE_ONNX: ${USE_ONNX}")
message(STATUS "USE_NLOHMANN_JSON: ${USE_NLOHMANN_JSON}")
message(STATUS "=============================")
message(STATUS "")
