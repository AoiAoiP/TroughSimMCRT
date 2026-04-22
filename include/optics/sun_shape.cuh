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

        float3 sun_local_dir = make_float3(
            sinf(theta) * cosf(phi),
            sinf(theta) * sinf(phi),
            -cosf(theta)
        );

        float s_zenith = config.zenith * PI / 180.0f;
        float s_azimuth = config.azimuth * PI / 180.0f;

        float3 sun_center = make_float3(
            sinf(s_zenith) * cosf(s_azimuth),
            sinf(s_zenith) * sinf(s_azimuth),
            -cosf(s_zenith)
        );

        return local2world(sun_local_dir, sun_center);
    }
}

