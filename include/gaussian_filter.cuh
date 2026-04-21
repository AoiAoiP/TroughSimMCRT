#pragma once

#include <cmath>
#include <vector>

class GaussianFilter {
private:
    std::vector<float> kernel;
    int radius;
    float sigma;

public:
    GaussianFilter(float sigma_val = 1.0f) : sigma(sigma_val) {
        radius = static_cast<int>(sigma * 3);
        computeKernel();
    }

    void computeKernel() {
        kernel.clear();
        float sum = 0.0f;
        
        // 计算高斯核
        for (int i = -radius; i <= radius; i++) {
            float val = std::exp(-0.5f * (i / sigma) * (i / sigma));
            kernel.push_back(val);
            sum += val;
        }
        
        // 归一化
        for (int i = 0; i < kernel.size(); i++) {
            kernel[i] /= sum;
        }
    }

    const std::vector<float>& getKernel() const { return kernel; }
    int getRadius() const { return radius; }
};

__device__ __forceinline__ int reflect(int x, int maxx)
{
    if (x < 0) return -x;
    if (x >= maxx) return (2 * maxx - x - 2);
    return x;
}

// 一维高斯滤波（水平方向）
__global__ void gaussian_blur_horizontal(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel,
    int radius
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int k = -radius; k <= radius; k++)
    {
        int xx = reflect(x + k, width);
        sum += input[y * width + xx] * kernel[k + radius];
    }

    output[y * width + x] = sum;
}

// 一维高斯滤波（垂直方向）
__global__ void gaussian_blur_vertical(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    const float* __restrict__ kernel,
    int radius
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;

    for (int k = -radius; k <= radius; k++)
    {
        int yy = reflect(y + k, height);
        sum += input[yy * width + x] * kernel[k + radius];
    }

    output[y * width + x] = sum;
}

// 二维高斯滤波（分离卷积）
void apply_gaussian_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    float sigma,
    cudaStream_t stream = 0
) {
    // 创建高斯滤波器
    GaussianFilter filter(sigma);
    const auto& kernel = filter.getKernel();
    int radius = filter.getRadius();
    
    // 分配设备内存存储核函数
    float* d_kernel;
    cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
    cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // 分配临时缓冲区
    float* d_temp;
    cudaMalloc(&d_temp, width * height * sizeof(float));
    
    // 设置线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // 水平滤波
    gaussian_blur_horizontal<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_temp, width, height, d_kernel, radius
    );
    
    // 垂直滤波
    gaussian_blur_vertical<<<gridSize, blockSize, 0, stream>>>(
        d_temp, d_output, width, height, d_kernel, radius
    );
    
    // 清理
    cudaFree(d_temp);
    cudaFree(d_kernel);
}