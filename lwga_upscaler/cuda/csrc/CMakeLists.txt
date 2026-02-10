cmake_minimum_required(VERSION 3.12)  # You can adjust the minimum required version

project(lwga_upscaler LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 75 89)  # Ti 2080 uses 75. V100 uses 70. RTX 4090 uses 89.

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# our library library
add_library(lwga_upscaler
    bindings.cu
    upscale.cu
    ext.cpp
)

target_link_libraries(lwga_upscaler PUBLIC CUDA::cudart pybind11::module)
target_include_directories(lwga_upscaler PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${pybind11_INCLUDE_DIRS}
)

# torch include (adjust path as needed)
# target_include_directories(lwga_upscaler PRIVATE path/to/torch/include)

target_include_directories(lwga_upscaler PRIVATE
        C:/Users/user/AppData/Local/pypoetry/Cache/virtualenvs/gsplat-sandbox-2l1tlsyc-py3.10/Lib/site-packages/torch/include
        C:/Users/user/AppData/Local/pypoetry/Cache/virtualenvs/gsplat-sandbox-2l1tlsyc-py3.10/Lib/site-packages/torch/include/torch/csrc/api/include
)
