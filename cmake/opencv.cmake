option(OpenCV_DIR "C:/opencv/" "Path to OpenCV lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})