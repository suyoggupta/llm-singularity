add_library(kernels STATIC
    src/cublas_provider.cpp
    src/attention.cu
    src/rmsnorm.cu
    src/activations.cu
    src/rope.cu
)
target_include_directories(kernels PUBLIC include)
target_link_libraries(kernels PUBLIC core CUDA::cublas CUDA::cudart)
