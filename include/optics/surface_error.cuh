#pragma once
#include "cu_math.cuh"
#include <curand_kernel.h>

namespace Optics{
    /**
     * @brief Apply Gaussian perturbation to a direction vector
     * @param vec Original direction (must be normalized, used as local Z axis)
     * @param sigma Standard deviation of the Gaussian angular perturbation (rad)
     * @param u1,u2 Two independent uniform random numbers in [0, 1)
     * @return Perturbed direction vector (normalized)
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