#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace PostProcess {

    #ifndef PI
    #define PI 3.14159265358979323846f
    #endif

    /**
     * @brief Gaussian filter configuration
     */
    struct GaussianConfig {
        int radius;    // filter kernel radius (e.g. radius=2 means 5x5 kernel)
        float sigma;   // Gaussian standard deviation
    };

    /**
     * @brief 2D Gaussian filter CUDA kernel
     * @param d_input_flux   Raw input flux density grid
     * @param d_output_flux  Smoothed output flux density grid
     * @param grid_res_x     Grid resolution along circumferential (X) direction
     * @param grid_res_z     Grid resolution along axial (Z) direction
     * @param config         Gaussian configuration
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

        // iterate over (2r+1) x (2r+1) Gaussian kernel neighbourhood
        for (int dz = -r; dz <= r; ++dz) {
            for (int dx = -r; dx <= r; ++dx) {

                // 1. Axial (Z) boundary: clamp to prevent crossing tube ends
                int neighbor_z = min(max(z + dz, 0), grid_res_z - 1);

                // 2. Circumferential (X) boundary: periodic wrap-around (cylinder seam)
                int neighbor_x = (x + dx) % grid_res_x;
                if (neighbor_x < 0) neighbor_x += grid_res_x;

                int neighbor_idx = neighbor_z * grid_res_x + neighbor_x;

                // 3. Compute Gaussian spatial weight
                // w = exp(-(dx^2 + dz^2) / (2 * sigma^2))
                float weight = expf(-(float)(dx * dx + dz * dz) / sigma22);

                sum_energy += d_input_flux[neighbor_idx] * weight;
                sum_weight += weight;
            }
        }

        // write normalized smoothed value to output grid
        int current_idx = z * grid_res_x + x;
        d_output_flux[current_idx] = sum_energy / sum_weight;
    }

    /**
     * @brief Host-side launch wrapper
     */
    void applyGaussianFilter(float* d_flux_map, int res_x, int res_z, int radius, float sigma) {
        
        size_t size = res_x * res_z * sizeof(float);
        
        // allocate temporary GPU buffer for smoothed result
        float* d_temp_flux;
        cudaMalloc((void**)&d_temp_flux, size);

        // configure 2D block/grid dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((res_x + blockSize.x - 1) / blockSize.x, 
                      (res_z + blockSize.y - 1) / blockSize.y);

        GaussianConfig config = {radius, sigma};

        // launch gaussian filter kernel
        gaussian_filter_2d_kernel<<<gridSize, blockSize>>>(d_flux_map, d_temp_flux, res_x, res_z, config);
        
        cudaDeviceSynchronize();

        // copy smoothed result back to original array
        cudaMemcpy(d_flux_map, d_temp_flux, size, cudaMemcpyDeviceToDevice);

        cudaFree(d_temp_flux);
    }

} // namespace PostProcess