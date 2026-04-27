#pragma once
#include "cu_math.cuh"
#include <curand_kernel.h>

namespace Optics{
    /**
     * @brief 通用的高斯扰动函数，对目标向量添加sigma高斯分布的随机扰动
     * @param vec 原始向量
     * @param sigma 高斯分布的标准差（rad）
     * @param state curand随机数状态
     * @return float3 扰动后的新向量
     */
    inline __device__ float3 GaussianPerturb(
        const float3& vec,
        float sigma,
        float u1,float u2
    ){
        if (sigma < 1e-6f) return vec;

        if(u1<1e-7f) u1=1e-7f;

        float theta = sigma * sqrtf(-2.0f * logf(u1));
        float phi = 2.0f * PI * u2;

        float sin_theta, cos_theta, sin_phi, cos_phi;
        sincosf(theta, &sin_theta, &cos_theta);
        sincosf(phi, &sin_phi, &cos_phi);

        float3 local_dir = make_float3(
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta
        );

        return local2world(local_dir,vec);
    }
}