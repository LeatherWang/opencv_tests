# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build

# Include any dependencies generated for this target.
include CMakeFiles/orb_match.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/orb_match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/orb_match.dir/flags.make

CMakeFiles/orb_match.dir/src/orb_match.cpp.o: CMakeFiles/orb_match.dir/flags.make
CMakeFiles/orb_match.dir/src/orb_match.cpp.o: ../src/orb_match.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/orb_match.dir/src/orb_match.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb_match.dir/src/orb_match.cpp.o -c /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/src/orb_match.cpp

CMakeFiles/orb_match.dir/src/orb_match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb_match.dir/src/orb_match.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/src/orb_match.cpp > CMakeFiles/orb_match.dir/src/orb_match.cpp.i

CMakeFiles/orb_match.dir/src/orb_match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb_match.dir/src/orb_match.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/src/orb_match.cpp -o CMakeFiles/orb_match.dir/src/orb_match.cpp.s

# Object files for target orb_match
orb_match_OBJECTS = \
"CMakeFiles/orb_match.dir/src/orb_match.cpp.o"

# External object files for target orb_match
orb_match_EXTERNAL_OBJECTS =

orb_match: CMakeFiles/orb_match.dir/src/orb_match.cpp.o
orb_match: CMakeFiles/orb_match.dir/build.make
orb_match: /usr/local/lib/libopencv_videostab.so.2.4.11
orb_match: /usr/local/lib/libopencv_ts.a
orb_match: /usr/local/lib/libopencv_superres.so.2.4.11
orb_match: /usr/local/lib/libopencv_stitching.so.2.4.11
orb_match: /usr/local/lib/libopencv_contrib.so.2.4.11
orb_match: /usr/local/lib/libopencv_nonfree.so.2.4.11
orb_match: /usr/local/lib/libopencv_ocl.so.2.4.11
orb_match: /usr/local/lib/libopencv_gpu.so.2.4.11
orb_match: /usr/local/lib/libopencv_photo.so.2.4.11
orb_match: /usr/local/lib/libopencv_objdetect.so.2.4.11
orb_match: /usr/local/lib/libopencv_legacy.so.2.4.11
orb_match: /usr/local/lib/libopencv_video.so.2.4.11
orb_match: /usr/local/lib/libopencv_ml.so.2.4.11
orb_match: /usr/local/lib/libopencv_calib3d.so.2.4.11
orb_match: /usr/local/lib/libopencv_features2d.so.2.4.11
orb_match: /usr/local/lib/libopencv_highgui.so.2.4.11
orb_match: /usr/local/lib/libopencv_imgproc.so.2.4.11
orb_match: /usr/local/lib/libopencv_flann.so.2.4.11
orb_match: /usr/local/lib/libopencv_core.so.2.4.11
orb_match: CMakeFiles/orb_match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable orb_match"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orb_match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/orb_match.dir/build: orb_match

.PHONY : CMakeFiles/orb_match.dir/build

CMakeFiles/orb_match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/orb_match.dir/cmake_clean.cmake
.PHONY : CMakeFiles/orb_match.dir/clean

CMakeFiles/orb_match.dir/depend:
	cd /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4 /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4 /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build /home/leather/lxdata/leather_repertory/leather_tests/opencv_test/harris_detect_2.4/build/CMakeFiles/orb_match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/orb_match.dir/depend
