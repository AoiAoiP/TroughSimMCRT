#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>
#include "config/json.hpp"
#include "cu_math.cuh"

enum class SunShapeType {
    UNIFORM,
    GAUSSIAN,
    BUIE,
    DEFINED
};

struct SunConfig {
    float azimuth;
    float zenith;
    float DNI;
    SunShapeType sunshape;

    union {
        float theta_max;    // UNIFORM
        float sigma;        // GAUSSIAN
    } params;

    int table_size;
    float* d_cdf_angles;     // 设备上的CDF角度（mrad）
    float* d_cdf_values;    // 设备上的CDF值
};

extern __constant__ SunConfig d_sun_config;

SunShapeType parseSunShapeType(const std::string& type_str);

SunConfig loadSunConfigToGPU(const std::string& filepath);

void printSunInfo(const SunConfig& config);