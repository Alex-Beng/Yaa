cmake_minimum_required(VERSION 3.10)
project(Yaa)

set(CMAKE_CXX_STANDARD 17)

link_directories(interception)
# 设置includepath
include_directories(interception)

add_subdirectory(apps)