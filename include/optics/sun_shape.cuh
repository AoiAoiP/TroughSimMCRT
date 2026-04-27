#pragma once
#include "config/sun_config.cuh"
#include "cu_math.cuh"
#include <curand_kernel.h>

namespace Optics{
    __device__ inline float SampleThetaFromcdf(float u,const SunConfig& config){
        // 二分查找CDF
        int left = 0;
        int right = config.table_size - 1;

        while (left < right - 1) {
            int mid = left + (right - left) / 2;
            if (config.d_cdf_values[mid] <= u) {
                left = mid;
            } else {
                right = mid;
            }
        }

        float p_low = config.d_cdf_values[left];
        float p_high = config.d_cdf_values[right];
        float t_low = config.d_cdf_angles[left];
        float t_high = config.d_cdf_angles[right];
        // 线性插值
        float t = (u - p_low) / (p_high - p_low);
        return (t_low + t * (t_high - t_low))*0.001f;
    }

    __device__ inline float3 SampleSunshape(float u1,float u2,const SunConfig& config){
        float theta = 0.0f;
        float phi = PI * 2.0f * u2;
        switch(config.sunshape){
            case SunShapeType::UNIFORM:
                theta = config.params.theta_max*0.001f * sqrtf(u1);
                break;
            case SunShapeType::GAUSSIAN:
                theta = sqrtf(-2.0f * logf(u1)) * config.params.sigma * 0.001f;
                break;
            case SunShapeType::BUIE:
                theta = SampleThetaFromcdf(u1, config);
                break;
            case SunShapeType::DEFINED:
                theta = SampleThetaFromcdf(u1, config);
                break;
        }

        float3 w = config.dir;
        float inv_len_w = rsqrtf(w.x*w.x + w.y*w.y + w.z*w.z);
        w.x *= inv_len_w; 
        w.y *= inv_len_w; 
        w.z *= inv_len_w;

        // 构建局部正交基底 (U, V, W)
        float3 a;
        if (fabsf(w.x) > 0.9f) {
            a = make_float3(0.0f, 1.0f, 0.0f);
        } else {
            a = make_float3(1.0f, 0.0f, 0.0f);
        }
        float3 u_vec = make_float3(
            a.y * w.z - a.z * w.y,
            a.z * w.x - a.x * w.z,
            a.x * w.y - a.y * w.x
        );
        float inv_len_u = rsqrtf(u_vec.x*u_vec.x + u_vec.y*u_vec.y + u_vec.z*u_vec.z);
        u_vec.x *= inv_len_u; 
        u_vec.y *= inv_len_u; 
        u_vec.z *= inv_len_u;
        float3 v_vec = make_float3(
            w.y * u_vec.z - w.z * u_vec.y,
            w.z * u_vec.x - w.x * u_vec.z,
            w.x * u_vec.y - w.y * u_vec.x
        );

        // 计算三角函数值 
        float sin_theta, cos_theta;
        sincosf(theta, &sin_theta, &cos_theta);
        float sin_phi, cos_phi;
        sincosf(phi, &sin_phi, &cos_phi);

        // 将局部坐标映射为全局 3D 射线向量 (等效于local2world)
        float3 global_dir = make_float3(
            sin_theta * cos_phi * u_vec.x + sin_theta * sin_phi * v_vec.x + cos_theta * w.x,
            sin_theta * cos_phi * u_vec.y + sin_theta * sin_phi * v_vec.y + cos_theta * w.y,
            sin_theta * cos_phi * u_vec.z + sin_theta * sin_phi * v_vec.z + cos_theta * w.z
        );

        return global_dir;
    }
}

