cmake_minimum_required(VERSION 3.16)
project(tool_classifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tool_classifier
    src/main.cpp
)

target_include_directories(tool_classifier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

