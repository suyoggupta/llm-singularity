add_executable(tests
    test_types.cpp
    test_memory.cpp
    test_sampling.cpp
    test_tokenizer.cpp
    test_kernels.cpp
    test_safetensors.cpp
    test_scheduler.cpp
    test_llama_model.cpp
    test_server.cpp
)
target_link_libraries(tests PRIVATE core kernels llama_model GTest::gtest_main)
add_test(NAME unit_tests COMMAND tests)

# Debug tool for forward pass comparison
add_executable(debug_dump debug_dump.cpp)
target_link_libraries(debug_dump PRIVATE core kernels llama_model)
