add_library(core STATIC
    src/server.cpp
    src/scheduler.cpp
    src/tokenizer.cpp
    src/sampling.cpp
    src/memory.cpp
)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC httplib::httplib nlohmann_json::nlohmann_json sentencepiece-static CUDA::cudart)
