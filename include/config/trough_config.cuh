#pragma once
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "config/json.hpp"

#define NUM_SUB_MIRRORS 6

struct TorsionErrorConfig{
    enum Type { NONE, POLYNOMIAL, LOOKUP};
    Type type;
    float coefficients[4];
    float* y_pos;
    float* torsion_values;
    int table_size;
};

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
    TorsionErrorConfig torsion_error;
};

extern __constant__ ParabolicTroughConfig d_trough_config;

ParabolicTroughConfig loadTroughConfigToGPU(const std::string& filepath);