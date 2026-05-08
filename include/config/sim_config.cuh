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
    unsigned int total_rays;    // total number of rays (e.g. 10^7)
    int block_size;             // CUDA thread block size (recommended 256 or 512)
    int rays_per_thread;        // rays per thread (for latency hiding)

    // absorber surface flux grid discretization
    int grid_res_z;             // grid divisions along axial direction (e.g. 100)
    int grid_res_x;             // grid divisions along circumferential direction (e.g. 36)

    unsigned long long seed;    // random seed
};

extern __constant__ SimConfig d_sim_config;

SimConfig loadSimConfigToGPU(const std::string& filepath);