cmake_minimum_required( VERSION 2.8 )
project( harris_detect_2.4 )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set(OpenCV_DIR "/home/leather/leather_software/opencv3.2_with_cuda_contrib/share/OpenCV")


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

find_package( OpenCV 2 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 

include_directories( 
    "./src/"
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/"
)

add_executable( orb_match ./src/orb_match.cpp  )
target_link_libraries( orb_match ${OpenCV_LIBS} )





