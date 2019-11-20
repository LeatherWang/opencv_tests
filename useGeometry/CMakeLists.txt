cmake_minimum_required( VERSION 2.8 )
project( geometry )

find_package(OpenCV REQUIRED)

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )

add_executable( eigenGeometry eigenGeometry.cpp )

target_link_libraries(eigenGeometry ${OpenCV_LIBS})
