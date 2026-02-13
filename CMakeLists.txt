cmake_minimum_required(VERSION 3.14)
project(SimplexMethod VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

include(FetchContent)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen3)

set(PROBLEM_TYPES_SOURCES
    src/ProblemTypes/Common.cpp
    src/ProblemTypes/Symmetrical.cpp
    src/ProblemTypes/Canonical.cpp
)

set(PARSER_SOURCES
    src/SymmetricalParser.cpp
)

add_library(simplex_lib STATIC
    ${PROBLEM_TYPES_SOURCES}
    ${PARSER_SOURCES}
)

target_include_directories(simplex_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(simplex_lib PUBLIC Eigen3::Eigen)

add_executable(simplex src/main.cpp)
target_link_libraries(simplex PRIVATE simplex_lib)

enable_testing()
add_subdirectory(tests)

option(BUILD_TESTS "Build tests" ON)
