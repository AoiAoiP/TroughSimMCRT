#include "optics/random_pools.cuh"
#include <random>
#include <iostream>

namespace Optics {
    __constant__ RandomPoolConfig d_pool_config;

    // 主机端指针记录，用于释放内存
    static float2* h_d_random_pool = nullptr;
    static int* h_d_rsia = nullptr;

    void initRandomPools(int pool_size, int num_rays) {

        // 1. 在 CPU 端分配临时内存
        float2* h_pool = new float2[pool_size];
        int* h_rsia = new int[num_rays];

        // 2. 使用 MT19937 生成高质量均匀分布随机数 [0.0, 1.0)
        std::mt19937 rng(12345); // 固定种子保证结果可复现
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> idx_dist(0, pool_size - 1);

        for (int i = 0; i < pool_size; ++i) {
            h_pool[i].x = dist(rng);
            h_pool[i].y = dist(rng);
        }

        for (int i = 0; i < num_rays; ++i) {
            h_rsia[i] = idx_dist(rng); // 每根光线分配一个随机起始偏移量
        }

        // 3. 分配 GPU 全局显存
        cudaMalloc(&h_d_random_pool, pool_size * sizeof(float2));
        cudaMalloc(&h_d_rsia, num_rays * sizeof(int));

        // 4. 数据拷贝至 GPU 全局内存
        cudaMemcpy(h_d_random_pool, h_pool, pool_size * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(h_d_rsia, h_rsia, num_rays * sizeof(int), cudaMemcpyHostToDevice);

        // 5. 组装配置结构体并拷贝至 __constant__ 内存
        RandomPoolConfig h_config;
        h_config.pool_size = pool_size;
        h_config.rsia_size = num_rays;
        h_config.d_random_pool = h_d_random_pool;
        h_config.d_rsia = h_d_rsia;

        cudaMemcpyToSymbol(d_pool_config, &h_config, sizeof(RandomPoolConfig));

        // 6. 清理 CPU 端临时内存
        delete[] h_pool;
        delete[] h_rsia;
        
        std::cout << "Successfully initialized Random Pools and RSIA." << std::endl;
    }

    void freeRandomPools() {
        if (h_d_random_pool) cudaFree(h_d_random_pool);
        if (h_d_rsia) cudaFree(h_d_rsia);
    }
}