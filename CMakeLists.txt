cmake_minimum_required(VERSION 4.0)
project(RoboArm)

set(CMAKE_CXX_STANDARD 20)

add_executable(RoboArm main.cpp
        tests/test.cpp)
