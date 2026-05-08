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

        // parameterize ray as P(t) = origin + t * dir
        // absorber tube is a cylinder: (P.x - pos.x)^2 + (P.z - pos.z)^2 = r^2
        // substituting yields quadratic: A*t^2 + B*t + C = 0
        float A = dir.x * dir.x + dir.z * dir.z;
        if(A < 1e-8f) {
            return rec; // ray parallel to absorber axis, no intersection
        }
        float B = 2.0f * ((origin.x - config.position.x) * dir.x + (origin.z - config.position.z) * dir.z);
        float C = (origin.x - config.position.x) * (origin.x - config.position.x) +
                  (origin.z - config.position.z) * (origin.z - config.position.z) -
                  config.r * config.r;

        // solve t
        float t_hit;
        if (!solve_quadratic(A, B, C, t_hit)) {
            return rec; // no intersection
        }

        // compute hit point P
        float3 P = make_float3(origin.x + t_hit * dir.x, origin.y + t_hit * dir.y, origin.z + t_hit * dir.z);

        // clip to absorber tube length
        if (P.y < config.position.y - config.length * 0.5f || P.y > config.position.y + config.length * 0.5f) {
            return rec; // outside the tube length
        }

        // surface normal for cylinder
        // F(x,y,z) = (x - pos.x)^2 + (z - pos.z)^2 - r^2 = 0
        // N = grad(F) = (2*(P.x - pos.x), 0, 2*(P.z - pos.z))
        float3 normal = make_float3(2.0f * (P.x - config.position.x), 0.0f, 2.0f * (P.z - config.position.z));
        normal = normalize(normal);

        if(dot(dir, normal) > 0.0f) {
            normal = -normal; // ensure normal faces the incoming ray
        }

        rec.is_hit = true;
        rec.t = t_hit;
        rec.hit_point = P;
        rec.normal = normal;
        rec.geometry_index = 6; // absorber tube geometry index

        return rec;
    }
} // namespace Geometry