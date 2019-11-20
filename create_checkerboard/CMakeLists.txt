cmake_minimum_required(VERSION 2.8)  
project(create_checkerboard_pro) 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")

find_package(OpenCV REQUIRED)    
add_executable(create_checkerboard create_checkerboard.cpp)    
target_link_libraries(create_checkerboard ${OpenCV_LIBS})    

add_executable(create_checkerboard_A4 create_checkerboard_A4.cpp)
target_link_libraries(create_checkerboard_A4 ${OpenCV_LIBS})
