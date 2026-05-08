<p align="right">
  <a href="README.md">English</a> | <a href="README_CN.md">简体中文</a>
</p>

# 槽式抛物面太阳能集热器 — CUDA 加速 MCRT 热仿真

基于 GPU 加速的高性能蒙特卡洛光线追踪（MCRT）求解器，用于计算槽式抛物面太阳能集热器（PTSC）集热管外表面的能流密度分布。

## 概述

本求解器采用**反向光线追踪**：光线从集热管出发，经抛物面镜反射后朝向太阳方向。只有实际到达集热管的光线才会被计入，因此计算高效且物理精度高。

## 核心特性

- **GPU 加速**：纯 CUDA C++ 实现，支持数千万至上亿条光线，仿真耗时毫秒级。
- **预生成随机池**：基于 MT19937 的随机 float2 数对和每光线随机起始索引（RSIA）在 CPU 端预计算并上传至 GPU 全局内存，避免 kernel 内部高频调用 `curand`。
- **精确几何建模**：
  - 鲁棒的一元二次方程求根公式，使用 `-0.5·(b ± √D)` 避免浮点精度相消。
  - 高效的 ray-paraboloid / ray-cylinder 求交运算（均为 inline）。
  - 支持 6 段子镜拼接及边界裁剪。
- **多模型太阳形状**：Uniform（均匀）、Gaussian（高斯）、Buie（CSR）以及通过 CSV 导入的自定义太阳模型。
- **高斯后处理**：5×5 高斯滤波核平滑蒙特卡洛噪声。圆周方向采用周期性边界（圆柱缝合），轴向采用 clamp 边界。
- **物理输出**：能流密度标定为 kW/m²，同时输出拦截因子与光学效率。

## 环境依赖

- **宿主编译器**: 支持 C++17 的 GCC 9.0+（或 MSVC）
- **CUDA Toolkit**: 11.0 或更高版本
- **CMake**: 3.18 或更高版本
- **Python**: 3.8+（用于可视化与参数扫描脚本）
- **Python 包**: `numpy`, `matplotlib`（年度仿真还需 `pandas`）

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/AoiAoiP/TroughSimMCRT.git
cd TroughSimMCRT

# 编译
mkdir build && cd build
cmake .. && make -j$(nproc)

# 运行单次仿真
./mcrt_sim

# 或一键运行完整管线（编译 + 仿真 + 绘图 + 分析）
cd .. && ./run.sh
```

仿真读取 `resources/config.json`，将能流密度 CSV 输出到 `out/` 目录。

## 仓库结构

```
.
├── CMakeLists.txt                  # CMake 构建配置 (CUDA SM 86)
├── run.sh                          # 一键运行脚本
├── README.md                       # 英文说明文档
├── README_CN.md                    # 中文说明文档
├── resources/
│   ├── config.json                 # 主仿真配置文件
│   ├── CSR-5.csv                   # Buie 太阳形状数据
│   └── 738_sundir_year.txt         # 全年太阳方向数据
├── include/
│   ├── app.cuh                     # 总入口头文件；SamplingDimension 枚举
│   ├── cu_math.cuh                 # CUDA 向量数学库 (float2/3/4 运算符重载)
│   ├── config/
│   │   ├── json.hpp                # nlohmann JSON 库 (单头文件)
│   │   ├── trough_config.cuh       # ParabolicTroughConfig 结构体 + 加载函数
│   │   ├── absorber_config.cuh     # AbsorberConfig 结构体 + 加载函数
│   │   ├── sun_config.cuh          # SunConfig 结构体 + 加载函数 + SunShapeType 枚举
│   │   ├── sim_config.cuh          # SimConfig 结构体 + 加载函数
│   │   └── sample_vMF.cuh          # von Mises-Fisher 分布采样（当前未被 kernel 使用）
│   ├── geometry/
│   │   ├── geometry_func.cuh       # HitInfo 结构体 + 鲁棒 solve_quadratic()
│   │   ├── trough_intersect.cuh    # sampleTrough() + intersectTrough()
│   │   └── absorber_intersect.cuh  # intersectAbsorber()
│   ├── optics/
│   │   ├── random_pools.cuh        # 随机池声明 + hash_index() + get_random_pair()
│   │   ├── sun_shape.cuh           # SampleSunshape() + SampleThetaFromcdf()
│   │   └── surface_error.cuh       # GaussianPerturb() (Box-Muller 变换)
│   └── postprocess/
│       └── gaussian_filter.cuh     # 5×5 高斯滤波 kernel + host 启动函数
├── src/
│   ├── main.cu                     # render() kernel + main() 管线
│   ├── trough_config.cu            # loadTroughConfigToGPU()
│   ├── sun_config.cu               # loadSunConfigToGPU()
│   ├── absorber_config.cu          # loadAbsorberConfigToGPU()
│   ├── sim_config.cu               # loadSimConfigToGPU()
│   └── random_pools.cu             # initRandomPools() + freeRandomPools()
└── scripts/
    ├── paint_energy_distribution.py # 2D 热力图 + 3D 表面图
    ├── scan_intercept.py            # 斜率误差 × 管径参数扫描
    ├── scan_focal_length.py         # 焦距敏感度扫描
    ├── find_optimal_focal.py        # 多开口宽度 × 焦距联合寻优
    └── annual_sim.py                # 全年能量加权拦截效率
```

## 配置参数说明 (`resources/config.json`)

### Simulation
| 键 | 说明 | 典型值 |
|---|---|--------|
| `total_rays` | 蒙特卡洛光线总数 | `10000000` |
| `block_size` | CUDA 线程块大小 | `256` |
| `grid_res_z` | 轴向网格数 | `100` |
| `grid_res_x` | 周向网格数 | `60` |

### ParabolicTrough
| 键 | 说明 | 典型值 |
|---|---|--------|
| `focal_length` | 焦距 (m) | `1.7` |
| `length` | 镜面长度 (m) | `1.8` |
| `width` | 毛开口宽度 (m) | `8.615` |
| `reflectivity` | 镜面反射率 | `0.92` |
| `slope_error` | RMS 面形斜率误差 (mrad) | `2.0` |
| `specularity_error` | RMS 镜面反射误差 (mrad) | `2.5` |
| `bounds` | 6 段子镜 X 区间 | 见配置文件 |

### Absorber
| 键 | 说明 | 典型值 |
|---|---|--------|
| `r` | 集热管半径 (m) | `0.045` |
| `position` | 管中心坐标 [X, Y, Z] (m) | `[0.019, 0.9, 1.7]` |
| `length` | 集热管长度 (m) | `1.9` |

### Sun
| 键 | 说明 | 典型值 |
|---|---|--------|
| `azimuth` / `zenith` | 太阳方位角/天顶角 (deg) | `180.0` / `0.0` |
| `dir` | 太阳方向向量 | `[0, 0, 1]` |
| `DNI` | 法向直射辐照度 (W/m²) | `1000.0` |
| `shape` | 太阳形状模型 | `UNIFORM`, `GAUSSIAN`, `BUIE`, `DEFINED` |
| `csv_path` | 自定义太阳形状 CSV 路径 | `../resources/CSR-5.csv` |

## 坐标系约定

- **X**: 抛物面开口宽度方向（水平横向），遵循 `z = x² / (4f)`
- **Y**: 镜面/集热管长度方向（轴向）
- **Z**: 竖直方向 / 抛物面深度方向

镜面中心位于 Y = 0.9 m；集热管中心位于 (X = 0.019, Y = 0.9, Z = 1.7) m。

## 物理模型

总等效表面误差计算公式：

```
σ_total = sqrt(4 × σ_slope² + σ_specularity²)  [mrad → rad: × 0.001]
```

能流标定公式：

```
flux [kW/m²] = hits × ρ_reflectivity × DNI × A_aperture / (N_rays × A_bin)
```

其中 `A_bin = 2πr × L_tube / (grid_res_x × grid_res_z)`。

周向分箱使用 `atan2(P.x - centre.x, P.z - centre.z) / 2π`，角度范围 [0, 2π)。

## 已知问题

- `main.cu` 第 117–125 行的能流汇总能量计算存在量纲错误：将 flux (kW/m²) 直接求和混同为能量值，`optical_efficiency` 的计算逻辑不严谨。
- `SampleThetaFromcdf` 中使用二分查找遍历 CDF 查找表，导致线程束内严重分支发散。可优化为均匀重采样查找表。
- `random_pools.cuh` 中 `hash_index` 的统计质量未经严格验证。
- Kernel 内缺少 `cudaDeviceSynchronize()`，无法捕获异步执行错误。

## 编译说明

- CUDA 目标架构：SM 86（RTX 30/40 系列）。如需适配其他 GPU，请修改 `CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES`。
- CUDA 编译器启用 `-use_fast_math`，以极微小的浮点精度损失换取超越函数（sin, cos, sqrt 等）的大幅性能提升。

## 开源协议

本项目用于学术研究。联系作者获取授权信息。
