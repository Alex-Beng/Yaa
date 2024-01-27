cmake_minimum_required(VERSION 3.10)


add_executable(mskb_test "ms_kb_test.cpp")
target_link_libraries(mskb_test interception)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)