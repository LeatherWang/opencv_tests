cmake_minimum_required( VERSION 2.8 )
project( feature_detect_ygz )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/home/leather/lxdata/leather_tools/opencv-3.4.1_with_cuda_contrib/share/OpenCV")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/library/libs/)

find_package(Boost 1.40 REQUIRED COMPONENTS filesystem system)

find_package( OpenCV 3 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 

include_directories( 
    "./include/"
    "./src/"
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/"
)




add_executable( sift_match ./src/sift_match.cpp  )
target_link_libraries( sift_match ${OpenCV_LIBS} )

add_executable( orb_match ./src/orb_match.cpp  )
target_link_libraries( orb_match ${OpenCV_LIBS} )


add_executable( lk_match ./src/lk_match.cpp  )
target_link_libraries( lk_match ${OpenCV_LIBS} )

add_executable( harris_detect ./src/harris_detect.cpp  )
target_link_libraries( harris_detect ${OpenCV_LIBS} )

add_library( shi_tomasi_api SHARED ./src/shi_tomasi_api.cpp  )
target_link_libraries( shi_tomasi_api
    ${OpenCV_LIBS})

AUX_SOURCE_DIRECTORY(./src/orb_slam_extractor SRC_ORB_SLAM_EXTRACTOR_API)
add_library(orb_slam_extractor SHARED ${SRC_ORB_SLAM_EXTRACTOR_API})
target_link_libraries( orb_slam_extractor
    ${OpenCV_LIBS}
    shi_tomasi_api)

add_executable( shi_tomasi ./src/shi_tomasi.cpp  )
target_link_libraries( shi_tomasi
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    shi_tomasi_api)

#add_executable( shi_tomasi_match ./src/shi_tomasi_match.cpp  )
#target_link_libraries( shi_tomasi_match
#    ${OpenCV_LIBS}
#    ${Boost_FILESYSTEM_LIBRARY}
#    ${Boost_SYSTEM_LIBRARY}
#    orb_slam_extractor)

#add_executable( shi_tomasi_match_stream ./src/shi_tomasi_match_stream.cpp  )
#target_link_libraries( shi_tomasi_match_stream
#    ${OpenCV_LIBS}
#    ${Boost_FILESYSTEM_LIBRARY}
#    ${Boost_SYSTEM_LIBRARY}
#    orb_slam_extractor)

add_executable( shi_tomasi_match_stream_vins ./src/shi_tomasi_match_stream_vins.cpp  )
target_link_libraries( shi_tomasi_match_stream_vins
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    orb_slam_extractor
    shi_tomasi_api)


add_executable( perspective_transformation ./src/perspective_transformation.cpp  )
target_link_libraries( perspective_transformation ${OpenCV_LIBS} )

add_executable( choose_quadrilateral ./src/choose_quadrilateral.cpp  )
target_link_libraries( choose_quadrilateral ${OpenCV_LIBS} )

add_executable( fast_imp ./src/fast_imp.cpp  )
target_link_libraries( fast_imp ${OpenCV_LIBS} )

add_executable( susan ./src/susan.cpp  )
target_link_libraries( susan
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    orb_slam_extractor)

# 将所有头文件拷贝到指定目录
file(GLOB akCompass_hdrs "${PROJECT_SOURCE_DIR}/include/*")
file(COPY ${akCompass_hdrs} DESTINATION "${PROJECT_SOURCE_DIR}/library/include/")

AUX_SOURCE_DIRECTORY(./src/line_descriptor SRCS_akVCompass)
add_library(LineDescriptorMe SHARED ${SRCS_akVCompass})
target_link_libraries( LineDescriptorMe ${OpenCV_LIBS} )

add_library( line_corners_api SHARED ./src/line_corners_api.cpp  )
target_link_libraries( line_corners_api
    ${OpenCV_LIBS}
    LineDescriptorMe)


add_executable( line_corners ./src/line_corners.cpp  )
target_link_libraries( line_corners
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    orb_slam_extractor
    line_corners_api)
