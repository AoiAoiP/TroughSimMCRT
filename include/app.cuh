#pragma once
#include "cu_math.cuh"
#include "config/trough_config.cuh"
#include "config/sun_config.cuh"
#include "config/absorber_config.cuh"
#include "config/sim_config.cuh"
#include "optics/sun_shape.cuh"
#include "optics/surface_error.cuh"
#include "optics/random_pools.cuh"
#include "geometry/trough_intersect.cuh"
#include "geometry/absorber_intersect.cuh"
#include "postprocess/gaussian_filter.cuh"

enum SamplingDimension {
    DIM_ORIGIN_GEN = 0,    // sample trough origin (u1, u2)
    DIM_SUN_SHAPE  = 1,    // sample sun cone direction (u3, u4)
    DIM_PERT_ERR  = 2,    // sample surface/specularity error perturbation (u5, u6)
};