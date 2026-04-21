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
     * @brief 在 CPU 端生成随机数（或 Halton 序列）并拷贝至 GPU 显存
     * @param pool_size 随机数对的数量
     * @param num_rays 仿真光线总数 (用于生成等长的 RSIA)
     */
    void initRandomPools(int pool_size = POOL_SIZE, int num_rays = 0);
    
    /**
     * @brief 释放显存
     */
    void freeRandomPools();

    /**
     * @brief 获取高质量的预计算随机数对 (核心查表函数)
     * @param ray_id 当前光线的全局 ID (tid)
     * @param dimension 当前采样的物理阶段 (0:起点采样, 1:太阳采样, 2:面形误差采样...)
     * @return float2 两个独立的 [0, 1) 均匀分布随机数
     */
    __device__ __forceinline__ float2 get_random_pair(int ray_id, int dimension) {
        
        int start_idx = d_pool_config.d_rsia[ray_id];
        unsigned int hashed_offset = hash_index(start_idx * 13 + dimension * 71);
        int pool_idx = hashed_offset & (d_pool_config.pool_size - 1);
        return d_pool_config.d_random_pool[pool_idx];
    }
} // namespace Optics