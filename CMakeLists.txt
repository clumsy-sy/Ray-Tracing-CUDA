cmake_minimum_required(VERSION 3.25)
project(RayTracing CUDA)

set(CMAKE_CUDA_STANDARD 20)

# aux_source_directory(src/external EXTERNAL)
# aux_source_directory(src/vector VECTOR)
add_executable(RayTracing src/main.cu )

set_target_properties(RayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
