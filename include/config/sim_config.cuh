#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>
#include "config/json.hpp"

struct SimConfig {
    unsigned int total_rays;    // 总光线数 (例如 10^7)
    int block_size;             // 线程块大小 (推荐 256 或 512)
    int rays_per_thread;        // 每个线程处理的光线数 (用于隐藏延迟)
    
    // 集热管表面能流网格划分
    int grid_res_z;             // 沿轴向的网格数 (例如 100)
    int grid_res_x;             // 沿周向(圆周)的网格数 (例如 36)
    
    unsigned long long seed;    // 随机数种子
};

extern __constant__ SimConfig d_sim_config;

SimConfig loadSimConfigToGPU(const std::string& filepath);