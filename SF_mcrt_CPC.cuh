#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <map>
#include <fstream>
#include "error.cuh"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

// #define BOUNCE_TEST
#define SLOPE_ERROR
#define RAND_STATE_POOL_SIZE 500000

constexpr float PI = 3.14159265358979323846f;
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BOUNCES = 50;
constexpr int ROUND_TIME = 10;
constexpr float EPSILON = 1e-6f;
constexpr float SURFACE_EPSILON = 1e-4f;
constexpr float solar_radius_rad = 0.00465f;

// 一次镜参数
constexpr int reflector_num = 20;//20
constexpr float reflector_len = 23.76f;
constexpr float reflector_width = 0.76f;
constexpr float reflector_reflectivity = 0.92f;
constexpr int Lpixel = 480;// 480
constexpr int Wpixel = 60; // 60
constexpr int reflector_spp = 30; // 30,一次镜上每个微元的采样数,一面镜子总共设定为864000根光线
constexpr int total_micro_pixels = reflector_num * Lpixel * Wpixel; // 总微元数576000
constexpr int total_rays_per_timestep = total_micro_pixels * reflector_spp; // 总光线数=镜数*长像素*宽像素*每像素微元采样数

constexpr float intv = 0.85f; // 镜子间距

// 二次镜参数
// *实际开口宽度为0.30m，高度为0.24m
constexpr float p0z = 8.1393f, lp1x = -0.075f, rp1x = 0.075f;
constexpr float p1z = 8.087f, lp2x = -0.15f, rp2x = 0.15f;
constexpr float p2z = 7.90f;
constexpr float specular_reflectivity = 0.96f;

// 吸收管参数
constexpr float receiver_R = 0.035f;
constexpr float receiver_h = 8.0f;
constexpr float receiver_len = 23.76f;
constexpr float receiver_reflectivity = 0.0f; // 吸收率为1
constexpr int M = 52;//theta方向分块数
constexpr int N = 23*42;//y方向分块数,966
constexpr int TOTAL_SEGMENTS = M * N;

// 玻璃管参数
constexpr float glass_len = 23.76f;
constexpr float glass_outer_R = 0.0625f;
constexpr float glass_inner_R = 0.0595f;
constexpr float glass_outer_h = 8.0f;
constexpr float glass_inner_h = 8.0f;
constexpr float coord_x = 0.0f;
constexpr float Ior = 1.523f;
constexpr float thickness = 0.002f;
constexpr float transmissivity = 0.95f;// 0.95f


// 虚拟平板参数
constexpr float plane_width = 0.3f; 
constexpr float plane_len = 23.76f;
constexpr float plane_z = 7.90f; 
constexpr float plane_center_x = 0.0f;
constexpr float plane_center_y = 0.0f;
constexpr int plane_Wpix = 60; // discretization along width
constexpr int plane_Lpix = 128; // discretization along length 

constexpr int HIT_NONE = 0;
constexpr int HIT_DIRECT = 1;
constexpr int HIT_SPECULAR = 2;

#ifdef BOUNCE_TEST
constexpr int NO_HIT_BOUNCE = -1;
#endif

#ifdef SLOPE_ERROR
constexpr float slope_error[10] = {0.0f, 0.0025f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f,0.5f,0.8f}; // 0, 2.5 mrad ,5mrad, 10mrad, 20mrad, 50mrad, 100mrad, 200mrad, 500mrad, 800mrad
#endif
