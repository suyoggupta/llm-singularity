add_executable(llm-serve-${MODEL} main.cpp)
target_link_libraries(llm-serve-${MODEL} PRIVATE core kernels llama_model)
