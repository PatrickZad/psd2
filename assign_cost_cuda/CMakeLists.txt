cmake_minimum_required(VERSION 3.6)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda-10.2/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(assign_cost_cuda SHARED
  assign_cost_cuda.cu)

target_link_libraries(assign_cost_cuda
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(assign_cost_cuda PROPERTIES PREFIX "")
set(CMAKE_BUILD_TYPE "Debug")