#pragma once
#include "geometry/geometry_func.cuh"
#include "config/trough_config.cuh"

namespace Geometry{
    // Sample a point uniformly over the 6-submirror trough surface
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

            // find which mirror segment the target falls in
            if (target_w < accumulated_w + segment_w) {
                // local fraction inside the current segment [0, 1]
                float local_u = (target_w - accumulated_w) / segment_w;

                // map to global x coordinate
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

    // Ray vs. segmented parabolic trough intersection
    // F(x,y,z) = x^2 - 4*f*z = 0  (parabolic cylinder along y)
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
            return rec; // no intersection
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
            normal = -normal; // ensure normal faces the incoming ray
        }

        // return value
        rec.is_hit = true;
        rec.t = t_hit;
        rec.hit_point = P;
        rec.normal = normal;

        return rec;
    }

    inline __device__ float computeTorsionAngle(float y, const TorsionErrorConfig& config){
        switch(config.type){
            case TorsionErrorConfig::NONE: return 0.0f;
            case TorsionErrorConfig::POLYNOMIAL:
                return (config.coefficients[0] + y * (config.coefficients[1] + y * config.coefficients[2])) * 0.001f;
            case TorsionErrorConfig::LOOKUP:
            {
                float* y_pos = config.y_pos;
                float* tors = config.torsion_values;
                int N = config.table_size;

                if (N < 2 || y <= y_pos[0]) return tors[0] * 0.001f;
                if (y >= y_pos[N - 1]) return tors[N - 1] * 0.001f;

                int lo = 0, hi = N - 1;
                while (hi - lo > 1) {
                    int mid = (lo + hi) >> 1;
                    if (y_pos[mid] <= y) lo = mid;
                    else hi = mid;
                }

                float t = (y - y_pos[lo]) / (y_pos[hi] - y_pos[lo]);
                return (tors[lo] + t * (tors[hi] - tors[lo])) * 0.001f;
            }
            default: return 0.0f;
        }
    }

} // namespace Geometry