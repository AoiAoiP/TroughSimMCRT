#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <cuda_runtime.h>
#include "config/json.hpp"

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
        float csr;          // BUIE
    } params;

    int table_size;
    float* d_cdf_angles;     // 设备上的CDF角度（mrad）
    float* d_cdf_values;    // 设备上的CDF值
};

extern __constant__ SunConfig d_sun_config;

SunShapeType parseSunShapeType(const std::string& type_str);

void loadSunConfigToGPU(const std::string& filepath);

void printSunInfo(const SunConfig& config);