cmake_minimum_required(VERSION 3.14)
project(Yaa)

set(CMAKE_CXX_STANDARD 17)
# in case Chinese note won't couse error
add_compile_options(/utf-8)

include(cmake/interception.cmake)
include(cmake/nlohmann_json.cmake)
include(cmake/opencv.cmake)
include(cmake/cli11.cmake)

add_subdirectory(apps)