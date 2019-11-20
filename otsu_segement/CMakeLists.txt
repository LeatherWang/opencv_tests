cmake_minimum_required(VERSION 2.8)  
project(otsu_segement) 

find_package(OpenCV REQUIRED)    
add_executable(otsu_seg otsu_seg.cpp)    
target_link_libraries(otsu_seg ${OpenCV_LIBS})    

