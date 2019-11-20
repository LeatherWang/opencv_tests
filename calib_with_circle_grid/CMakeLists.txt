cmake_minimum_required( VERSION 2.8 )
project( calib_with_circle_grid )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR "/home/leather/lxdata/leather_tools/opencv-3.1.0/share/OpenCV")

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.4 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
find_package(Boost 1.40 REQUIRED COMPONENTS filesystem system)

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

add_executable( choose_color choose_color.cpp  )
target_link_libraries( choose_color ${OpenCV_LIBS} )

add_executable( create_asymmetrical_circle_grid create_asymmetrical_circle_grid.cpp  )
target_link_libraries( create_asymmetrical_circle_grid ${OpenCV_LIBS} )

add_executable( circle_grid_detect circle_grid_detect.cpp  )
target_link_libraries( circle_grid_detect ${OpenCV_LIBS} )

add_executable( choose_quadrilateral choose_quadrilateral.cpp  )
target_link_libraries( choose_quadrilateral ${OpenCV_LIBS} )

add_executable( choose_fisheye_mask choose_fisheye_mask.cpp  )
target_link_libraries( choose_fisheye_mask ${OpenCV_LIBS} )

add_executable( calib_with_circle_grid calib_with_circle_grid.cpp  )
target_link_libraries( calib_with_circle_grid ${OpenCV_LIBS} )

add_executable( calib_with_checkerboard_fisheye calib_with_checkerboard_fisheye.cpp  )
target_link_libraries( calib_with_checkerboard_fisheye
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})

add_executable( calib_with_circle_grid_fisheye calib_with_circle_grid_fisheye.cpp  )
target_link_libraries( calib_with_circle_grid_fisheye
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})

add_executable( calib_with_circle_grid_general calib_with_circle_grid_general.cpp  )
target_link_libraries( calib_with_circle_grid_general
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})

add_executable( fisheye_calibration_circle fisheye_calibration_circle.cpp  )
target_link_libraries( fisheye_calibration_circle
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})


add_executable( choose_blob choose_blob.cpp  )
target_link_libraries( choose_blob
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})

add_executable( test_fov test_fov.cpp  )
target_link_libraries( test_fov
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})

add_executable( video_to_image video_to_image.cpp  )
target_link_libraries( video_to_image
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY})
