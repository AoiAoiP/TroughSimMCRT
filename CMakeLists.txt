cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Parabolic_MCRT_Sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# CUDA target architecture:
#  75 : RTX 20 series,  80 : A100 / RTX 30 series,  86 : RTX 30/40 series
set(CMAKE_CUDA_ARCHITECTURES 86)
# ---------------------------------------------------------

set(MCRT_SOURCES
    src/main.cu
    src/trough_config.cu
    src/sun_config.cu
    src/absorber_config.cu
    src/sim_config.cu
    src/random_pools.cu
)

set(MCRT_HEADERS
    include/cu_math.cuh
    include/config/trough_config.cuh
    include/config/absorber_config.cuh
    include/config/sun_config.cuh
    include/config/sim_config.cuh
    include/optics/sun_shape.cuh
    include/optics/surface_error.cuh
    include/optics/random_pools.cuh
    include/geometry/geometry_func.cuh
    include/geometry/absorber_intersect.cuh
    include/geometry/trough_intersect.cuh
    include/postprocess/gaussian_filter.cuh
    include/app.cuh
)

add_executable(mcrt_sim ${MCRT_SOURCES} ${MCRT_HEADERS})

set_target_properties(mcrt_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set header include path to source directory
target_include_directories(mcrt_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# Per-language compile optimisation options
target_compile_options(mcrt_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3                   # Max optimisation level
        -use_fast_math        # Trade tiny precision loss for much faster math (sin, cos, etc.)
        --generate-line-info  # Keep line number info for Nsight Compute profiling
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)