# 使用CUDA-MCRT的槽式CSP系统热仿真
## 项目介绍
本项目是一个基于 CUDA 加速 的蒙特卡洛光线追踪（MCRT）热仿真求解器，专门针对槽式抛物面太阳能集热器（Parabolic Trough Solar Collector, PTSC）的集热管外表面能流密度分布进行高精度、高并发的计算分析。

## 核心特性
- 使用高性能 GPU 并行计算，设计了纯 CUDA C++ 实现双向光线追踪算法，支持千万级至亿级光线的秒级仿真。
- 引入预生成的微定日镜法线池 (MHNP) 和太阳形状池 (SSP)，有效避免了在 Kernel 函数中高频调用随机数生成器（如 `curand`），大幅提升计算效率与精度。
- 精准的几何建模：
- - 支持一元二次方程的鲁棒求根公式（解决浮点数精度相消问题）。
- - 抛物面与圆柱面（集热管）的高效极致内联求交逻辑。
- - 支持多块子镜拼接的边界判定拦截。
- 多模型太阳形状 (Sun Shape)：支持 Uniform、Gaussian、Buie (CSR) 太阳模型建模，也支持自定义太阳模型导入。

## 环境依赖
在编译和运行本项目之前，请确保您的系统已安装以下环境：
- **编译器**: 支持 C++17 的宿主编译器 (GCC 9.0+ 或 MSVC)
- **CUDA Toolkit**: 11.0 或更高版本
- **CMake**: 3.18 及以上
- **Python**: Python 3.8+ (仅用于运行可视化脚本，需安装 `numpy`, `matplotlib`, `pandas`)

## 编译与构建
本项目采用标准的 CMake 构建系统，在终端中执行以下命令进行编译：
```
git clone https://github.com/AoiAoiP/TroughSimMCRT.git
cd TroughSimMCRT

mkdir build && cd build

cmake ..

make -j8
```

## 运行与可视化
编译成功后，可执行文件将生成在 build 目录下。
### 1. 运行仿真核心
```
# 在 build 目录下运行
./TroughSimMCRT
```
程序将自动读取 `resources/config.json` 中的参数，开始执行 MCRT 仿真，并在完成后将能流密度数据输出到 CSV 文件中。
### 2. 结果可视化
仿真结束后，可以通过提供的 Python 脚本将 CSV 文件渲染为热力图/能流分布图。
```
# 返回项目根目录
cd ..
# 运行可视化脚本 (请确保已安装必要的 Python 绘图库)
python scripts/paint_energy_distribution.py
```

### 3. 一步到位（Linux环境）
如果是Linux环境，直接运行`run.sh`即可。
```
# 直接运行一键脚本
./run.sh
```

## 参数配置说明
本项目的核心仿真参数通过 `resources/config.json` 进行管理，修改此文件无需重新编译代码。核心配置项说明：
- `sim_config`: 设置总光线数、CUDA Block Size、Grid 策略等。
- `trough_config`: 设置抛物面的焦距、长度、宽度、镜面误差及子镜拼接边界。
- `absorber_config`: 集热管的几何参数（如内径、长度尺寸、空间坐标等）。
- `sun_shape`: 选择太阳模型（Buie/Gaussian等）以及对应模型的误差角。