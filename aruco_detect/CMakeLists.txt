cmake_minimum_required( VERSION 2.8 )
project( aruco_detect_project )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/home/leather/leather_software/opencv3.2_with_cuda_contrib/share/OpenCV")

# 设置二进制的输出路径和库文件的输出路径
#set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
#set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.2 REQUIRED )
        
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories(
    ##将src目录下的头文件加入
    ${CMAKE_CURRENT_SOURCE_DIR} #当前处理的 CMakeLists.txt 所在的路径
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

#AUX_SOURCE_DIRECTORY(./src/hello SRC_DIR_HELLO)

#ADD_LIBRARY(aruco_detect_project SHARED
#    ${SRC_DIR_HELLO}
#)

##一次性将需要用到的库放到一起使用，使用新名称:LINK_LIBS
LIST(APPEND LINK_LIBS
    #aruco_detect_project
    ${OpenCV_LIBS}
)

add_executable( detect_maker ./src/detect_maker.cpp  )
target_link_libraries( detect_maker ${LINK_LIBS} )

add_executable( detect_board ./src/detect_board.cpp  )
target_link_libraries( detect_board ${LINK_LIBS} )

add_executable( create_board ./src/create_board.cpp  )
target_link_libraries( create_board ${LINK_LIBS} )

add_executable( create_marker ./src/create_marker.cpp  )
target_link_libraries( create_marker ${LINK_LIBS} )
