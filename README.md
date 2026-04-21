# 使用CUDA加速的基于双向光线追踪的线性菲涅尔系统的集热管热仿真
## 项目结构
Parabolic_Sim/
├── CMakeLists.txt                  # 现代构建系统入口，指定C++17标准和CUDA架构
├── README.md                       # 项目说明文档
├── scripts/                        # 存放所有非编译类的脚本文件
│   └── paint_energy_distribution.py # Python 可视化出图脚本
|
├── resources/                       # 存放所有的输入输出资源文件
│   ├── config.json                  # 仿真参数配置文件
|   └── CSR-5.csv                   # 预生成的太阳形状数据（CSR-5模型），CSV格式
│
├── include/                        # 存放所有的头文件 (.cuh / .h)，纯数据定义和函数声明
│   ├── config/                     # 【配置与数据层】彻底取代原来的 SF_mcrt_CPC.cuh
│   │   ├── sim_config.cuh          # 宏观仿真参数：光线总数、Block/Grid Size、环境参数等
│   │   ├── trough_config.cuh       # POD结构体：抛物面焦距、长宽，以及6面子镜的边界数组
│   │   └── absorber_config.cuh     # POD结构体：单层集热管的半径、长度、表面光学属性
│   │
│   ├── math/                       # 【数学与底层工具层】
│   │   ├── helper_math.h           # NVIDIA原生的向量运算库，替代手写的vector.cuh
│   │   └── root_solver.cuh         # 鲁棒的一元二次方程求根公式（解决精度相消问题）
│   │
│   ├── geometry/                   # 【几何求交逻辑层】纯数学计算模块
│   │   ├── trough_intersect.cuh    # 声明：光线与抛物面的局部坐标系求交函数
│   │   └── absorber_intersect.cuh  # 声明：光线与圆柱集热管的求交函数
│   │
│   ├── optics/                     # 【光学与随机数抽样层】
│   │   ├── sun_shape.cuh           # 声明：Buie、Gaussian模型等太阳锥采样逻辑
│   │   └── random_pools.cuh        # 声明：QMCRT核心的 SSP 和 MHNP 显存池加载与读取接口
│   │
│   └── postprocess/                # 【后处理与滤波层】
│       ├── tms_filter.cuh          # 声明：截尾均值平滑(TMS)滤波算法
│       └── gaussian_filter.cuh     # 声明：高斯滤波算法
│
└── src/                            # 存放所有的实现文件 (.cu / .cpp)
    ├── geometry/
    │   ├── trough_intersect.cu     # 实现：极致内联的光线与统一数学抛物面求交，及6面边界判定
    │   └── absorber_intersect.cu   # 实现：集热管圆柱面求交与吸收判断
    │
    ├── optics/
    │   ├── sun_shape.cu            # 实现：根据配置文件在主机/设备端生成太阳形状数据
    │   └── random_pools.cu         # 实现：预生成 微定日镜法线池(MHNP) 和 太阳形状池(SSP)
    │
    ├── postprocess/
    │   ├── tms_filter.cu           # 实现：基于共享内存(Shared Memory)的TMS滤波 Kernel
    │   └── file_io.cu              # 实现：将吸收能流密度数组 SaveAsCSV 到外部存储
    │
    └── main.cu                     # 【主干控制流】组装各个模块，绝不写具体的数学求交逻辑