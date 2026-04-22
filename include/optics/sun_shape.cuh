#pragma once
#include "config/sun_config.cuh"
#include "cu_math.cuh"
#include <curand_kernel.h>

namespace Optics{
    /*
        @brief Buie太阳采样
        @param local_state 随机数状态
        @param csr 太阳半径（弧度）
        @return 太阳光线方向
    */
    __device__ inline float SampleBuieSunshape(float u,float csr) {
        const float theta_disk = 4.65e-3f;
        const float theta_aureole = 9.3e-3f;
        
        // 限制csr范围
        csr = fmaxf(csr, 0.05f);
        csr = fminf(csr, 0.5f);
        
        float theta_buie;
        float rand_val = u;
        
        // 根据面积比例选择区域，避免拒绝采样
        if (rand_val < 0.98f) { // 约98%的光线在太阳盘内
            // 太阳盘：均匀采样
            theta_buie = u * theta_disk;
        } else {
            // 光晕：使用近似的高斯分布
            float mean_theta = (theta_disk + theta_aureole) * 0.5f;
            float sigma = (theta_aureole - theta_disk) * 0.5f;
            theta_buie = fabsf(u) * sigma + mean_theta;
            theta_buie = fminf(theta_buie, theta_aureole);
        }
        
        // 随机选择正负角度
        if (u < 0.5f) {
            theta_buie = -theta_buie;
        }

        return theta_buie;
    }

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
        if(config.d_cdf_values == nullptr || config.d_cdf_angles == nullptr) return make_float3(0.0f,0.0f,-1.0f); 

        float theta = 0.0f;
        float phi = PI * 2.0f * u2;
        switch(config.sunshape){
            case SunShapeType::UNIFORM:
                theta = config.params.theta_max * sqrtf(u1);
                break;
            case SunShapeType::GAUSSIAN:
                theta = sqrtf(-2.0f * logf(u1)) * config.params.sigma;
                break;
            case SunShapeType::BUIE:
                theta = SampleBuieSunshape(u1, config.params.csr);
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

