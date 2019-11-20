cmake_minimum_required( VERSION 2.8 )
project( mymat_pro )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

include_directories( 
    
)

add_executable(mymat ./mymat.cpp)

