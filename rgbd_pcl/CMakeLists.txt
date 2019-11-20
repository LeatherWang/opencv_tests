cmake_minimum_required( VERSION 2.8 )
project( rgbd_pcl )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/home/leather/leather_tools/opencv3.2_with_contrib/share/OpenCV")
set(OpenCV_INCLUDE_DIRS "/home/leather/leather_tools/opencv3.2_with_contrib/include/")
set(OpenCV_LIBS "/home/leather/leather_tools/opencv3.2_with_contrib/lib/")

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories(
    ##将src目录下的头文件加入
    ${CMAKE_CURRENT_SOURCE_DIR} #什么作用?
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

##一次性将需要用到的库放到一起使用，使用新名称:LINK_LIBS
LIST(APPEND LINK_LIBS
    ${OpenCV_LIBS}
)


add_executable( detect ./src/detect.cpp  )
target_link_libraries( detect ${LINK_LIBS} )

