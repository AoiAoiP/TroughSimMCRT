#pragma once
#include "SF_mcrt_CPC.cuh"

struct vec3f {
    float x, y, z;
};

// --------------------向量计算部分--------------------
/*
    @brief 计算向量的点积
    @param a 向量a
    @param b 向量b
    @return 点积结果
*/
__host__ __device__ __forceinline__ float dot3(const vec3f &a, const vec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*
    @brief 计算向量的加法
    @param a 向量a
    @param b 向量b
    @return 加法结果
*/
__device__ __forceinline__ vec3f add3(const vec3f &a, const vec3f &b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

/*
    @brief 计算向量的减法
    @param a 向量a
    @param b 向量b
    @return 减法结果
*/
__host__ __device__ __forceinline__ vec3f sub3(const vec3f &a, const vec3f &b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

/*
    @brief 计算向量的叉积
    @param a 向量a
    @param b 向量b
    @return 叉积结果
*/
__device__ __forceinline__ vec3f cross3(const vec3f &a, const vec3f &b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ __forceinline__ vec3f mul3(const vec3f &a, float s)
{
    return {a.x * s, a.y * s, a.z * s};
}

__host__ __device__ __forceinline__ vec3f normalize3(const vec3f &v)
{
    float len = sqrtf(dot3(v, v));
    if (len > 0)
    {
        return {v.x / len, v.y / len, v.z / len};
    }
    return {0, 0, 0};
}

/*
    @brief 计算两个点的距离
    @param v 向量v
    @return 向量长度
*/
__host__ __device__ __forceinline__ float distance3(const vec3f &a, const vec3f &b)
{
    return sqrtf(dot3(sub3(a, b), sub3(a, b)));
}

/*
    @brief 计算反射向量
    @param I 入射向量
    @param N 法线向量
    @return 反射向量
*/
__device__ __forceinline__ vec3f reflect3(const vec3f &I, const vec3f &N)
{
    float k = 2.0f * dot3(I, N);
    return {I.x - k * N.x, I.y - k * N.y, I.z - k * N.z};
}

/*
    @brief 对法线添加高斯分布斜率误差扰动
    @param n 原始法线
    @param rng 随机数状态
    @param sigma_rad 斜率误差标准差（弧度）
    @return 扰动后的法线
*/
__device__ __forceinline__ vec3f perturb_N(const vec3f &n, curandState *rng, float sigma_rad)
{
    // 找一个与n正交的基底 (t1, t2)
    vec3f t1;
    if (fabsf(n.x) > fabsf(n.z))
    {
        t1 = vec3f{-n.y, n.x, 0.0f};
    }
    else
    {
        t1 = vec3f{0.0f, -n.z, n.y};
    }
    t1 = normalize3(t1);
    vec3f t2 = normalize3(cross3(n, t1));

    // 生成高斯扰动
    float d1 = sigma_rad * curand_normal(rng);
    float d2 = sigma_rad * curand_normal(rng);

    // 添加扰动并归一化
    vec3f n_pert = normalize3(add3(n, add3(mul3(t1, d1), mul3(t2, d2)))); // n + d1*t1 + d2*t2
    return n_pert;
}

/*
    @brief 计算点p绕Y轴从x0y平面旋转到法向量为Normal的平面时的位置
    @param p 点p
    @param Normal 新平面法向量
    @return 新点p
*/
__host__ __device__ __forceinline__ vec3f rotate3(const vec3f &p, const vec3f &Normal){
    float theta = atan2f(Normal.x, Normal.z);  // 计算旋转角度
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    return {p.x * cos_theta + p.z * sin_theta, p.y, - p.x * sin_theta + p.z * cos_theta};
}