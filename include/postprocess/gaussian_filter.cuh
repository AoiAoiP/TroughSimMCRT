#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace PostProcess {

    #ifndef PI
    #define PI 3.14159265358979323846f
    #endif

    /**
     * @brief 高斯滤波配置参数
     */
    struct GaussianConfig {
        int radius;    // 滤波半径 (例如 radius=2 意味着 5x5 的卷积核)
        float sigma;   // 高斯分布标准差
    };

    /**
     * @brief 执行 2D 高斯滤波的 CUDA Kernel
     * * @param d_input_flux   原始输入能流密度网格
     * @param d_output_flux  平滑后的能流密度网格
     * @param grid_res_x     X方向(环向)网格数
     * @param grid_res_z     Z方向(轴向)网格数
     * @param config         高斯配置参数
     */
    __global__ void gaussian_filter_2d_kernel(
        const float* d_input_flux,
        float* d_output_flux,
        int grid_res_x,
        int grid_res_z,
        GaussianConfig config)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int z = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= grid_res_x || z >= grid_res_z) return;

        float sum_energy = 0.0f;
        float sum_weight = 0.0f;

        int r = config.radius;
        float sigma = config.sigma;
        float sigma22 = 2.0f * sigma * sigma;

        // 遍历 (2r+1) x (2r+1) 的高斯核邻域
        for (int dz = -r; dz <= r; ++dz) {
            for (int dx = -r; dx <= r; ++dx) {
                
                // 1. Z 轴 (轴向) 处理：Clamp 边界保护 (不越过管子两端)
                int neighbor_z = min(max(z + dz, 0), grid_res_z - 1);

                // 2. X 轴 (环向) 处理：Periodic 周期循环边界 (圆柱面缝合)
                // 巧妙利用取模和加法，处理 dx 为负数的情况
                int neighbor_x = (x + dx) % grid_res_x;
                if (neighbor_x < 0) neighbor_x += grid_res_x;

                int neighbor_idx = neighbor_z * grid_res_x + neighbor_x;

                // 3. 计算高斯空间权重
                // w = exp(-(dx^2 + dz^2) / (2 * sigma^2))
                float weight = expf(-(float)(dx * dx + dz * dz) / sigma22);

                sum_energy += d_input_flux[neighbor_idx] * weight;
                sum_weight += weight;
            }
        }

        // 将归一化后的平滑值写入新网格
        int current_idx = z * grid_res_x + x;
        d_output_flux[current_idx] = sum_energy / sum_weight;
    }

    /**
     * @brief 供主机端调用的启动函数
     */
    void applyGaussianFilter(float* d_flux_map, int res_x, int res_z, int radius, float sigma) {
        
        size_t size = res_x * res_z * sizeof(float);
        
        // 申请一块临时显存用于存放平滑结果
        float* d_temp_flux;
        cudaMalloc((void**)&d_temp_flux, size);

        // 设置 2D Block 维度
        dim3 blockSize(16, 16);
        dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, 
                      (res_z + blockSize.y - 1) / blockSize.y);

        GaussianConfig config = {radius, sigma};

        // 启动高斯滤波 Kernel
        gaussian_filter_2d_kernel<<<gridSize, blockSize>>>(d_flux_map, d_temp_flux, res_x, res_z, config);
        
        cudaDeviceSynchronize();

        // 将平滑后的数据拷回原数组
        cudaMemcpy(d_flux_map, d_temp_flux, size, cudaMemcpyDeviceToDevice);

        cudaFree(d_temp_flux);
    }

} // namespace PostProcess