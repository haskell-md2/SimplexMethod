cmake_minimum_required(VERSION 3.20)
project(SimplexMethod)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/include/eigen3)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(main ${SOURCES}
        src/ProblemTypes/Canonical.cpp
        src/ProblemTypes/Canonical.cpp)
