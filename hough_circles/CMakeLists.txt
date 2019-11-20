cmake_minimum_required(VERSION 2.8)  
project(hough_circles) 
   
find_package(OpenCV REQUIRED)    
add_executable(hough_circles hough_circles.cpp)    
target_link_libraries(hough_circles ${OpenCV_LIBS})    

