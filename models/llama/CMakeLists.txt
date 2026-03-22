add_library(llama_model STATIC
    llama_model.cpp
    llama_runtime.cpp
)
target_include_directories(llama_model PUBLIC ${CMAKE_SOURCE_DIR}/models/include)
target_link_libraries(llama_model PUBLIC core kernels nlohmann_json::nlohmann_json)
