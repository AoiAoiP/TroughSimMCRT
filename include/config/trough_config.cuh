#pragma once
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "config/json.hpp"

#define NUM_SUB_MIRRORS 6

struct ParabolicTroughConfig {
    float focal_length;
    float length;
    float3 position;
    float width;
    float valid_width;

    float2 bounds[NUM_SUB_MIRRORS]; 

    float reflectivity;
    float slope_error;
    float specularity_error;
};

extern __constant__ ParabolicTroughConfig d_trough_config;

void loadTroughConfigToGPU(const std::string& filepath);