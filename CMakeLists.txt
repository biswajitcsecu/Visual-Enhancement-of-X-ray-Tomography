
cmake_minimum_required(VERSION 3.9)

project(houghlines )
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)
FIND_PACKAGE(ArrayFire REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(houghlines houghlines.cpp)
target_link_libraries(houghlines ${OpenCV_LIBS} ${ArrayFire_LIBRARIES})



