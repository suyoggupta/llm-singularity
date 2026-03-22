add_library(core STATIC
    src/server.cpp
    src/scheduler.cpp
    src/tokenizer.cpp
    src/sampling.cpp
    src/memory.cpp
    src/safetensors.cpp
)
target_include_directories(core PUBLIC include)
target_include_directories(core PRIVATE
    ${sentencepiece_SOURCE_DIR}/src
)
target_link_libraries(core PUBLIC httplib::httplib nlohmann_json::nlohmann_json sentencepiece-static CUDA::cudart)
