cmake_minimum_required(VERSION 3.10)
project(MolecularDynamics LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)  # Enforce C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)  # CUDA standard remains 14

add_executable(molecular_dynamics
  src/main.cu
  src/particle.cu
  input/cli.cu
  src/vtk_writer.cu  # Now a .cu file
)

target_link_libraries(molecular_dynamics PRIVATE stdc++fs)  # For older GCC

set_target_properties(molecular_dynamics PROPERTIES
  CUDA_ARCHITECTURES "89"
  CUDA_SEPARABLE_COMPILATION ON
)