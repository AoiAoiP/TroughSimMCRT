#pragma once
#include <curand_kernel.h>

namespace Optics {
    constexpr int POOL_SIZE = 1048576; // 1024 * 1024

    struct RandomPoolConfig {
        int pool_size;
        int rsia_size;

        float2* d_random_pool;
        int* d_rsia;
    };

    extern __constant__ RandomPoolConfig d_pool_config;

    inline __device__ unsigned int hash_index(unsigned int seed) {
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
    }

    /**
     * @brief Generate random numbers on CPU and upload to GPU memory
     * @param pool_size Number of random float2 pairs in the pool
     * @param num_rays Total ray count (for generating equal-length RSIA array)
     */
    void initRandomPools(int pool_size = POOL_SIZE, int num_rays = 0);

    /**
     * @brief Free GPU memory
     */
    void freeRandomPools();

    /**
     * @brief Fetch a precomputed random float2 pair (core lookup function)
     * @param ray_id Current ray's global thread ID
     * @param dimension Sampling stage (0: origin, 1: sun shape, 2: surface error)
     * @return float2 Two independent uniform random values in [0, 1)
     */
    __device__ __forceinline__ float2 get_random_pair(int ray_id, int dimension) {
        
        int start_idx = d_pool_config.d_rsia[ray_id];
        unsigned int hashed_offset = hash_index(start_idx * 13 + dimension * 71);
        int pool_idx = hashed_offset & (d_pool_config.pool_size - 1);
        return d_pool_config.d_random_pool[pool_idx];
    }
} // namespace Optics