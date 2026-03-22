add_executable(tests
    test_types.cpp
    test_memory.cpp
    test_sampling.cpp
    test_tokenizer.cpp
    test_kernels.cpp
    test_safetensors.cpp
    test_scheduler.cpp
)
target_link_libraries(tests PRIVATE core kernels GTest::gtest_main)
add_test(NAME unit_tests COMMAND tests)
