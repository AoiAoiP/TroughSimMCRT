cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 定义项目名称，并显式启用 C++ 和 CUDA 语言支持
project(Parabolic_MCRT_Sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------
# CUDA 架构设置, 指定生成代码的 GPU 架构.
#  75 :RTX 20 系，80 :A100/RTX 30 系，86:RTX 30/40 系
set(CMAKE_CUDA_ARCHITECTURES 86)
# ---------------------------------------------------------


# 收集所有的 CUDA 实现文件 (.cu)
set(MCRT_SOURCES
    src/main.cu
    src/trough_config.cu
    src/sun_config.cu
    src/absorber_config.cu
    src/sim_config.cu
    src/random_pools.cu
)

# 收集所有的头文件 (.cuh)
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

# 创建可执行文件目标
add_executable(mcrt_sim ${MCRT_SOURCES} ${MCRT_HEADERS})

set_target_properties(mcrt_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 设置头文件包含路径为当前源码目录
target_include_directories(mcrt_sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# 针对不同语言配置特定的编译优化选项
target_compile_options(mcrt_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3                   # 启用最高级别的优化
        -use_fast_math        # 牺牲极微小的精度换取大量数学函数（如 sin, cos）的性能提升
        --generate-line-info  # 保留行号信息，便于配合 Nsight Compute 进行性能分析，且不影响运行速度
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
    >
)