#pragma once
#include "geometry/geometry_func.cuh"
#include "config/absorber_config.cuh"

namespace Geometry{
    inline __device__ HitInfo intersectAbsorber(
        const float3& origin,
        const float3& dir,
        const AbsorberConfig& config
    ){
        HitInfo rec;
        rec.is_hit = false;
        rec.geometry_index = -1;

        // 将光线参数化为 P(t) = origin + t * dir
        // 集热管是一个圆柱面，满足 (P.x - config.position.x)^2 + (P.z - config.position.z)^2 = config.r^2
        // 代入参数化方程得到二次方程：A*t^2 + B*t + C = 0
        float A = dir.x * dir.x + dir.z * dir.z;
        if(A < 1e-8f) {
            return rec; // 光线平行于集热管轴线，无交点
        }
        float B = 2.0f * ((origin.x - config.position.x) * dir.x + (origin.z - config.position.z) * dir.z);
        float C = (origin.x - config.position.x) * (origin.x - config.position.x) +
                  (origin.z - config.position.z) * (origin.z - config.position.z) -
                  config.r * config.r;

        // solve t 
        float t_hit;
        if (!solve_quadratic(A, B, C, t_hit)) {
            return rec; // 无交点
        }

        // 计算交点 P
        float3 P = make_float3(origin.x + t_hit * dir.x, origin.y + t_hit * dir.y, origin.z + t_hit * dir.z);

        // 判断交点是否在集热管长度范围内
        if (P.y < config.position.y - config.length * 0.5f || P.y > config.position.y + config.length * 0.5f) {
            return rec; // 不在长度范围内
        }
        
        // calculate normal
        // F(x,y,z) = (x - config.position.x)^2 + (z - config.position.z)^2 - config.r^2 = 0
        // N = grad(F) = (2*(P.x - config.position.x), 0, 2*(P.z - config.position.z))
        float3 normal = make_float3(2.0f * (P.x - config.position.x), 0.0f, 2.0f * (P.z - config.position.z));
        normal = normalize(normal);

        if(dot(dir, normal) > 0.0f) {
            normal = -normal; // 确保法线朝向入射光线
        }

        rec.is_hit = true;
        rec.t = t_hit;
        rec.hit_point = P;
        rec.normal = normal;
        rec.geometry_index = 6; // 集热管的几何索引

        return rec;
    }
} // namespace Geometry