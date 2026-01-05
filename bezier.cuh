#pragma once
#include "vector.cuh"

// --------------------Bezier部分--------------------
/*
    @brief 计算二次贝塞尔曲线
    @param p0 控制点0
    @param p1 控制点1
    @param p2 控制点2
    @param t 参数t
    @return 贝塞尔曲线值
*/
__host__ __device__ __forceinline__ float bezier_curve(float p0, float p1, float p2, float t)
{
    return (1.0f - t) * (1.0f - t) * p0 + 2.0f * t * (1.0f - t) * p1 + t * t * p2;
}

/*
    @brief 计算二次贝塞尔曲线的导数
    @param p0 控制点0
    @param p1 控制点1
    @param p2 控制点2
    @param t 参数t
    @return 贝塞尔曲线导数值
*/
__device__ __forceinline__ float bezier_curve_derivative(float p0, float p1, float p2, float t)
{
    return 2.0f * (1.0f - t) * (p1 - p0) + 2.0f * t * (p2 - p1);
}

/*
    @brief 计算光线与2D Bezier(x,z)的交点
    @param origin 光线起点
    @param dir 光线方向
    @param p0x 贝塞尔曲线控制点0的x坐标
    @param p0z 贝塞尔曲线控制点0的z坐标
    @param p1x 贝塞尔曲线控制点1的x坐标
    @param p1z 贝塞尔曲线控制点1的z坐标
    @param p2x 贝塞尔曲线控制点2的x坐标
    @param p2z 贝塞尔曲线控制点2的z坐标
    @param t_bezier_best 最佳t_bezier值
    @param hit_x 命中点的x坐标
    @param hit_z 命中点的z坐标
    @param t_ray_best 最佳t_ray值
    @return 是否命中
*/
__device__ bool hit_bezier(
    const vec3f &origin,
    const vec3f &dir,
    float p0x, float p0z,
    float p1x, float p1z,
    float p2x, float p2z,
    float &t_bezier_best,
    float &hit_x, float &hit_z,
    float &t_ray_best)
{
    // 计算二次方程系数
    float a = p0z - 2.0f * p1z + p2z;
    float b = 2.0f * (p1z - p0z);
    float c = p0z - origin.z;

    float d = p0x - 2.0f * p1x + p2x;
    float e = 2.0f * (p1x - p0x);
    float f = p0x - origin.x;

    // 构建关于t的二次方程: A*t^2 + B*t + C = 0
    float A = a * dir.x - d * dir.z;
    float B = b * dir.x - e * dir.z;
    float C = c * dir.x - f * dir.z;

    bool ishit = false;
    t_ray_best = 1e30f;
    t_bezier_best = 0.0f;

    // 处理退化情况（A=0，退化为线性方程）
    if (fabsf(A) < 1e-8f)
    {
        if (fabsf(B) > 1e-12f)
        {
            float t = -C / B;
            if (0.0f <= t <= 1.0f)
            {
                float z = bezier_curve(p0z, p1z, p2z, t);
                float x = bezier_curve(p0z, p1x, p2x, t);
                // 检查光线方向是否朝向交点
                float t_ray = (fabsf(dir.z) > 1e-12f) ? (z - origin.z) / dir.z
                                                      : (x - origin.x) / dir.x;
                if (t_ray > 0.0f)
                {
                    ishit = true;
                    t_ray_best = t_ray;
                    t_bezier_best = t;
                    hit_x = x;
                    hit_z = z;
                }
            }
        }
    }
    else
    {
        // 正常二次方程求解
        float discriminant = B * B - 4.0f * A * C;
        if (discriminant >= 0.0f)
        {
            float sqrt_disc = sqrtf(discriminant);
            float t1 = (-B + sqrt_disc) / (2.0f * A);
            float t2 = (-B - sqrt_disc) / (2.0f * A);
            // 尝试两个解，选择最近的正 t_ray
            auto try_root = [&](float t)
            {
                if (t < 0.0f || t > 1.0f)
                    return;
                float z = bezier_curve(p0z, p1z, p2z, t);
                float x = bezier_curve(p0x, p1x, p2x, t);
                float t_ray = (fabsf(dir.z) > 1e-12f) ? (z - origin.z) / dir.z
                                                      : (x - origin.x) / dir.x;
                if (t_ray > 0.0f && t_ray < t_ray_best)
                {
                    ishit = true;
                    t_bezier_best = t;
                    hit_x = x;
                    hit_z = z;
                    t_ray_best = t_ray;
                }
            };
            try_root(t1);
            try_root(t2);
        }
    }
    return ishit;
}