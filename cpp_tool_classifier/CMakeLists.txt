cmake_minimum_required(VERSION 3.15)
project(ToolClassifier VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(nlohmann_json 3.11.0 QUIET)

# If nlohmann_json is not found, use FetchContent to download it
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Add executable
add_executable(tool_classifier tool_classifier.cpp)

# Link libraries
target_link_libraries(tool_classifier PRIVATE nlohmann_json::nlohmann_json)

# Compiler warnings
if(MSVC)
    target_compile_options(tool_classifier PRIVATE /W4)
else()
    target_compile_options(tool_classifier PRIVATE -Wall -Wextra -pedantic)
endif()

# Installation
install(TARGETS tool_classifier DESTINATION bin)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
