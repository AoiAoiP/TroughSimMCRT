#pragma once
#include "geometry/geometry_func.cuh"
#include "config/trough_config.cuh"

namespace Geometry{
    /*
    * @brief 根据随机数采样镜面表面点
    */
    inline __device__ float3 sampleTrough(float u1,float u2,
        const ParabolicTroughConfig& config,
        int& mirror_id
    ){
        float target_w = u1 * config.valid_width;
        float accumulated_w = 0.0f;
        float x = 0.0f;
        mirror_id = 0;
        #pragma unroll
        for(int i = 0;i<NUM_SUB_MIRRORS;++i){
            float segment_w = config.bounds[i].y - config.bounds[i].x;
        
            // 如果 target 落在当前子镜的累加宽度范围内
            if (target_w < accumulated_w + segment_w) {
                // 计算在当前子镜内部的局部随机比例 (0.0 到 1.0 之间)
                float local_u = (target_w - accumulated_w) / segment_w;
                
                // 映射到真实的全局 x 坐标
                x = config.bounds[i].x + local_u * segment_w;
                mirror_id = i;
                break; 
            }
            accumulated_w += segment_w;
        }

        float y = (u2 - 0.5f)*config.length + config.position.y;
        float z = (x*x)/(4.0f * config.focal_length);
        return make_float3(x,y,z);
    }

    /*
    * @brief 光线与分段槽式抛物面核心求交函数
    * @param origin 光线起点
    * @param dir 光线方向（必须归一化）
    * @param config 抛物面参数配置（in 常量内存）
    * @return HitInfo 相交结果
    */
    inline __device__ HitInfo intersectTrough(
        const float3& origin,
        const float3& dir,
        const ParabolicTroughConfig& config
    ){
        HitInfo rec;
        rec.is_hit = false;
        rec.geometry_index = -1;

        // construct quadratic coefficients for ray-paraboloid intersection
        float f = config.focal_length;
        float a = dir.x * dir.x;
        float b = 2.0f *origin.x * dir.x - 4.0f * f * dir.z;
        float c = origin.x * origin.x - 4.0f * f * origin.z;

        // solve t
        float t_hit;
        if(!solve_quadratic(a, b, c, t_hit)) {
            return rec; // 无交点
        }

        // calculate hit point
        float3 P = make_float3(origin.x + t_hit * dir.x, origin.y + t_hit * dir.y, origin.z + t_hit * dir.z);

        // clip the length
        if(P.y < config.position.y - config.length * 0.5f || P.y > config.position.y + config.length * 0.5f)
            return rec;
        
        // check the boards of the trough
        bool in_segment = false;
        #pragma unroll
        for (int i = 0;i< NUM_SUB_MIRRORS;i++){
            if(P.x >= config.bounds[i].x && P.x <= config.bounds[i].y){
                in_segment = true;
                rec.geometry_index = i;
                break;
            }
        }
        if(!in_segment)
            return rec;

        // calculate the normal of P
        // F(x,y,z) = x^2 -4*f*z = 0
        // N = (2x,0,-4f)
        float3 normal = make_float3(2.0f*P.x,0.0f,-4.0f*f);
        normal = normalize(normal);

        if(dot(dir, normal) > 0.0f) {
            normal = -normal; // 确保法线朝向入射光线
        }

        // return value
        rec.is_hit = true;
        rec.t = t_hit;
        rec.hit_point = P;
        rec.normal = normal;

        return rec;
    }

} // namespace Geometry