add_executable(tests
    test_types.cpp
)
target_link_libraries(tests PRIVATE core kernels GTest::gtest_main)
add_test(NAME unit_tests COMMAND tests)
