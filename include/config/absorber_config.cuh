#pragma once
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "config/json.hpp"

struct AbsorberConfig {
    float r;
    float3 position;
    float length;
};

extern __constant__ AbsorberConfig d_absorber_config;

void loadAbsorberConfigToGPU(const std::string& filepath);