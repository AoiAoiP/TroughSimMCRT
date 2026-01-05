#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
// #include <cub/cub.cuh>

class TMSFilter {
private:
    int kernel_size;
    float trim_ratio;  // 修剪比例 p%

public:
    TMSFilter(int size = 11, float ratio = 0.03f) : kernel_size(size), trim_ratio(ratio) {}
    
    int getKernelSize() const { return kernel_size; }
    float getTrimRatio() const { return trim_ratio; }
    int getRadius() const { return kernel_size / 2; }
};

__device__ __host__ inline int clamp(int x, int min_val, int max_val) {
    return (x < min_val) ? min_val : ((x > max_val) ? max_val : x);
}

// 堆排序的辅助函数
__device__ void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ void heapify(float* arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && arr[left] > arr[largest])
        largest = left;
    
    if (right < n && arr[right] > arr[largest])
        largest = right;
    
    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// 使用堆选择算法在共享内存中实现部分排序
__global__ void trimmed_mean_smoothing_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    int kernel_radius,
    float trim_ratio
) {
    extern __shared__ float shared_memory[];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int kernel_size = 2 * kernel_radius + 1;
    int total_pixels = kernel_size * kernel_size;
    int trim_count = static_cast<int>(total_pixels * trim_ratio);
    int valid_count = total_pixels - 2 * trim_count;
    
    // 收集邻域像素到共享内存
    float* neighborhood = shared_memory + threadIdx.y * blockDim.x * total_pixels 
                         + threadIdx.x * total_pixels;
    
    int idx = 0;
    for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
            int y = clamp(row + dy, 0, height - 1);
            int x = clamp(col + dx, 0, width - 1);
            neighborhood[idx++] = input[y * width + x];
        }
    }
    
    __syncthreads();
    
    // 使用堆排序对邻域数组进行排序
    for (int i = total_pixels / 2 - 1; i >= 0; i--) {
        heapify(neighborhood, total_pixels, i);
    }
    
    for (int i = total_pixels - 1; i >= 0; i--) {
        swap(neighborhood[0], neighborhood[i]);
        heapify(neighborhood, i, 0);
    }
    
    __syncthreads();
    
    // 计算修剪后的均值
    float sum = 0.0f;
    for (int i = trim_count; i < total_pixels - trim_count; i++) {
        sum += neighborhood[i];
    }
    
    output[row * width + col] = sum / valid_count;
}

// 优化的 TMS 实现（使用 warp-level 操作）
__global__ void trimmed_mean_smoothing_optimized(
    const float* input,
    float* output,
    int width,
    int height,
    int kernel_radius,
    float trim_ratio
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int kernel_size = 2 * kernel_radius + 1;
    int total_pixels = kernel_size * kernel_size;
    int trim_count = static_cast<int>(total_pixels * trim_ratio);
    
    // 使用寄存器存储邻域值（对小核适用）
    float neighborhood[121];  // 最大支持 11x11 核
    
    // 收集邻域像素
    int idx = 0;
    for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
        for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
            int y = clamp(row + dy, 0, height - 1);
            int x = clamp(col + dx, 0, width - 1);
            neighborhood[idx++] = input[y * width + x];
        }
    }
    
    // 部分排序：使用插入排序（对小数组高效）
    for (int i = 1; i < total_pixels; i++) {
        float key = neighborhood[i];
        int j = i - 1;
        while (j >= 0 && neighborhood[j] > key) {
            neighborhood[j + 1] = neighborhood[j];
            j--;
        }
        neighborhood[j + 1] = key;
    }
    
    // 计算修剪后的均值
    float sum = 0.0f;
    for (int i = trim_count; i < total_pixels - trim_count; i++) {
        sum += neighborhood[i];
    }
    
    output[row * width + col] = sum / (total_pixels - 2 * trim_count);
}


/*
    @brief 主 TMS 应用函数
    @param 输入
    @param 输出
    @param 宽度
    @param 长度
    @param 卷积核大小
    @param 裁剪比例，默认3%
*/
void apply_trimmed_mean_smoothing(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int kernel_size = 11,
    float trim_ratio = 0.03f,  // 默认修剪 3%
    cudaStream_t stream = 0
) {
    TMSFilter filter(kernel_size, trim_ratio);
    int radius = filter.getRadius();
    
    // 验证参数
    if (kernel_size % 2 == 0) {
        printf("Warning: Kernel size should be odd, using %d instead\n", kernel_size + 1);
        kernel_size += 1;
    }
    
    if (kernel_size * kernel_size > 121) {
        printf("Error: Kernel size too large for optimized implementation\n");
        return;
    }
    
    // 设置线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // 计算共享内存需求
    size_t shared_mem_size = blockSize.x * blockSize.y * kernel_size * kernel_size * sizeof(float) 
                           + 1024;  // 额外空间给 CUB
    
    // 根据核大小选择实现
    if (kernel_size <= 11) {
        // 对小核使用优化版本（无共享内存）
        trimmed_mean_smoothing_optimized<<<gridSize, blockSize, 0, stream>>>(
            d_input, d_output, width, height, radius, trim_ratio
        );
    } else {
        // 对大核使用共享内存版本(堆排序)
        trimmed_mean_smoothing_kernel<<<gridSize, blockSize, shared_mem_size, stream>>>(
            d_input, d_output, width, height, radius, trim_ratio
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in TMS: %s\n", cudaGetErrorString(err));
        printf("kernel size:%d ,Grid size: (%d,%d), Block size: (%d,%d)\n", 
            kernel_size,gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    }
}

// 保留原高斯滤波接口的兼容版本
void apply_tms_filter(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    float trim_ratio = 0.03f,
    cudaStream_t stream = 0
) {
    // 使用Duan论文中推荐的 11x11 核
    apply_trimmed_mean_smoothing(d_input, d_output, width, height, 11, trim_ratio, stream);
}