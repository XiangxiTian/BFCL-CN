cmake_minimum_required(VERSION 3.16)
project(tool_taxonomy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED)
find_package(CURL REQUIRED)

add_executable(tool_taxonomy
    src/main.cpp
    src/utils.cpp
    src/tool_loader.cpp
    src/embedder.cpp
    src/options.cpp
    src/kmeans.cpp
)

target_include_directories(tool_taxonomy
    PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tool_taxonomy
    PRIVATE
        CURL::libcurl
)
