cmake_minimum_required(VERSION 3.10)
project(YOLOV8)

set (OpenCV_DIR C:/build/build/install/x64/vc17/lib)

find_package(OpenCV REQUIRED)  
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(YOLOV8 main.cpp)  # Change "main.cpp" par ton fichier source principal

target_link_libraries(YOLOV8 ${OpenCV_LIBS} ${CUDA_LIBRARIES})