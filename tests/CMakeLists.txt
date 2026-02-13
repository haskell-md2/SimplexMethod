include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(TEST_SOURCES
    test_common.cpp
    test_symmetrical.cpp
    test_canonical.cpp
    test_parser.cpp
    test_transformations.cpp
)

add_executable(run_tests ${TEST_SOURCES})

target_link_libraries(run_tests
    PRIVATE
    simplex_lib
    GTest::gtest_main
)

gtest_discover_tests(run_tests)
