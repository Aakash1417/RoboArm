cmake_minimum_required(VERSION 3.15)
project(CameraTest CXX)

find_package(realsense2 REQUIRED)  # realsense2::realsense2

add_executable(CameraTest
        src/main.cpp
)

target_link_libraries(CameraTest PRIVATE realsense2::realsense2)
set_property(TARGET CameraTest PROPERTY CXX_STANDARD 17)
