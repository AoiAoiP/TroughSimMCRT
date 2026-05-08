#include "optics/random_pools.cuh"
#include <random>
#include <iostream>

namespace Optics {
    __constant__ RandomPoolConfig d_pool_config;

    // host-side pointer record for freeing GPU memory later
    static float2* h_d_random_pool = nullptr;
    static int* h_d_rsia = nullptr;

    void initRandomPools(int pool_size, int num_rays) {

        // 1. Allocate temporary CPU memory
        float2* h_pool = new float2[pool_size];
        int* h_rsia = new int[num_rays];

        // 2. Generate high-quality uniform random numbers via MT19937 [0.0, 1.0)
        std::mt19937 rng(12345); // fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::uniform_int_distribution<int> idx_dist(0, pool_size - 1);

        for (int i = 0; i < pool_size; ++i) {
            h_pool[i].x = dist(rng);
            h_pool[i].y = dist(rng);
        }

        for (int i = 0; i < num_rays; ++i) {
            h_rsia[i] = idx_dist(rng); // each ray gets a random starting index into the pool
        }

        // 3. Allocate GPU global memory
        cudaMalloc(&h_d_random_pool, pool_size * sizeof(float2));
        cudaMalloc(&h_d_rsia, num_rays * sizeof(int));

        // 4. Copy data to GPU global memory
        cudaMemcpy(h_d_random_pool, h_pool, pool_size * sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(h_d_rsia, h_rsia, num_rays * sizeof(int), cudaMemcpyHostToDevice);

        // 5. Assemble config struct and copy to __constant__ memory
        RandomPoolConfig h_config;
        h_config.pool_size = pool_size;
        h_config.rsia_size = num_rays;
        h_config.d_random_pool = h_d_random_pool;
        h_config.d_rsia = h_d_rsia;

        cudaMemcpyToSymbol(d_pool_config, &h_config, sizeof(RandomPoolConfig));

        // 6. Free CPU-side temporary memory
        delete[] h_pool;
        delete[] h_rsia;

        std::cout << "Successfully initialized Random Pools and RSIA." << std::endl;
    }

    void freeRandomPools() {
        if (h_d_random_pool) cudaFree(h_d_random_pool);
        if (h_d_rsia) cudaFree(h_d_rsia);
    }
}