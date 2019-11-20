cmake_minimum_required( VERSION 2.8 )
project( cuda_orb_pro )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( OpenCV_DIR "/home/leather/leather_tools/opencv3.2_with_cuda_contrib/share/OpenCV")

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

add_executable( cuda_orb cuda_orb.cpp  )
target_link_libraries( cuda_orb ${OpenCV_LIBS} )

