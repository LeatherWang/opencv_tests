cmake_minimum_required(VERSION 2.8)  
project(create_charuco_pro) 

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")

find_package(OpenCV 3.4 REQUIRED)

add_executable(create_charuco create_charuco.cpp)    
target_link_libraries(create_charuco ${OpenCV_LIBS})    

add_executable(detect_aruco detect_aruco.cpp)
target_link_libraries(detect_aruco ${OpenCV_LIBS})

