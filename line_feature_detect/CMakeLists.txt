cmake_minimum_required(VERSION 2.8)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
        MESSAGE("CCACHE_FOUND! " )
endif(CCACHE_FOUND)


project(LineFeatureDetect)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
include(cotire)

#/home/leather/leather_software/opencv3.2_with_cuda_contrib/share/OpenCV
set(OpenCV_DIR "/home/leather/lxdata/leather_tools/opencv-3.1.0/share/OpenCV")

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

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

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

include_directories( /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/line_descriptor/include/opencv2)
link_directories(/home/leather/lxdata/leather_repertory/leather_tests/opencv_test/line_descriptor/build)


add_executable(line_feature ./src/line_feature.cpp)
target_link_libraries(line_feature
    ${OpenCV_LIBS}
    line_descriptor
    )


add_executable(line_feature_new ./src/line_feature_new.cpp)
target_link_libraries(line_feature_new
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    line_descriptor
)
cotire(line_feature_new)

add_executable(line_feature_new_once ./src/line_feature_new_once.cpp)
target_link_libraries(line_feature_new_once
    ${OpenCV_LIBS}
    line_descriptor
    )


#add_executable(line_feature_optimize ./src/line_feature_optimize.cpp)
#target_link_libraries(line_feature_optimize
#    ${OpenCV_LIBS}
#    ${Boost_FILESYSTEM_LIBRARY}
#    ${Boost_SYSTEM_LIBRARY}
#    ${Boost_THREAD_LIBRARY}
#    line_descriptor
#)


