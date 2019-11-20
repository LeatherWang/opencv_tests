cmake_minimum_required( VERSION 2.8 )
project( histogram_equalization_pro )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/home/leather/lxdata/leather_tools/opencv-3.1.0/share/OpenCV")

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

##### Boost #####
# Expand the next statement if newer boost versions than 1.40.0 are released
# set(Boost_ADDITIONAL_VERSIONS "1.40" "1.40.0" "1.49" "1.49.0")
find_package(Boost 1.40 REQUIRED COMPONENTS filesystem program_options serialization system thread)
# MSVC seems to be the only compiler requiring date_time
if(MSVC)
  find_package(Boost 1.40 REQUIRED date_time)
endif(MSVC)
# No auto linking, so this option is useless anyway
mark_as_advanced(Boost_LIB_DIAGNOSTIC_DEFINITIONS)


find_package( OpenCV 3.1 REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/"
)

add_executable( histogram_equalization histogram_equalization.cpp  )
target_link_libraries( histogram_equalization
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    )

