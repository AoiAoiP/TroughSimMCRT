#pragma once
#include "cu_math.cuh"

namespace Geometry{
    struct HitInfo {
        bool is_hit;            // whether the ray intersects geometry
        float t;                // hit distance along ray
        float3 hit_point;       // hit position in world space
        float3 normal;          // surface normal at hit point (normalized)
        int geometry_index;     // geometry index: 0-5 = mirror segment, 6 = absorber tube, -1 = no hit
    };

    // Robust quadratic root solver: returns the smallest t > 1e-4
    // Uses -0.5*(b ± sqrt(discriminant)) to avoid catastrophic cancellation
    inline __device__ bool solve_quadratic(float a, float b, float c, float& t) {
        if(fabsf(a) < 1e-6f) { // degenerate to linear equation
            if (fabsf(b) < 1e-6f) return false; // no solution
            t = -c / b;
            return t > 1e-4f;
        }

        float discriminant = b * b - 4.0f * a * c;
        if (discriminant < 0.0f) return false; // no real roots

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