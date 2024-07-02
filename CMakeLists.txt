cmake_minimum_required(VERSION 3.14)
project(Yaa)

set(CMAKE_CXX_STANDARD 17)
# in case Chinese note won't couse error
add_compile_options(/utf-8)
# 设置bin目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

# use interception de
option(ENABLE_INTERCEPTION "enable interception, which need to install driver on your pc" OFF)

if(ENABLE_INTERCEPTION)
    include(cmake/interception.cmake)
    add_definitions(-DENABLE_INTERCEPTION)
endif()
include(cmake/nlohmann_json.cmake)
include(cmake/opencv.cmake)
include(cmake/cli11.cmake)

add_subdirectory(apps)