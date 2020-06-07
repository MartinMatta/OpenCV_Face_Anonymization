cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(Face_anonymization)

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 11)