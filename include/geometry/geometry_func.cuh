#pragma once
#include "cu_math.cuh"

namespace Geometry{
    struct HitInfo {
        bool is_hit;        // 是否与几何体相交
        float t;            // 交点距离
        float3 hit_point;   // 交点坐标
        float3 normal;      // 交点法线(归一化)
        int geometry_index;   // 交点所在的几何索引，镜面为0-5，集热管为6，未击中为-1
    };

    /*
    * @brief 一元二次方程求根函数
    */
    inline __device__ bool solve_quadratic(float a, float b, float c, float& t) {
        if(fabsf(a) < 1e-6f) { // 退化为线性方程
            if (fabsf(b) < 1e-6f) return false; // 无解
            t = -c / b;
            return t > 1e-4f;
        }

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) return false; // 无实根

        float sqrt_disc = sqrtf(discriminant);
        float q = (b > 0.0f) ? -0.5f * (b + sqrt_disc) : -0.5f * (b - sqrt_disc);
        float t1 = q / a;
        float t2 = c / q;
        if(t1>t2) {float tmp = t1; t1 = t2; t2 = tmp;}
        if(t1 > 1e-4f) {
            t = t1;
            return true;
        }
        if(t2 > 1e-4f) {
            t = t2;
            return true;
        }
        
        return false;
    }
}