cmake_minimum_required(VERSION 3.14)


add_executable(mskb_test "ms_kb_test.cpp")
target_link_libraries(mskb_test interception)

add_executable(jsonl_test "jsonl_test.cpp")
target_link_libraries(jsonl_test interception nlohmann_json::nlohmann_json)


add_executable(mskb_recorder "ms_kb_recorder.cpp")
target_link_libraries(mskb_recorder interception nlohmann_json::nlohmann_json)
