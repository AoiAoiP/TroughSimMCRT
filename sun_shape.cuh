#pragma once
#include "SF_mcrt_CPC.cuh"
#include "vector.cuh"

enum shape {
    GAUSSIAN,
    BUIE,
    UNIFORM
};

class Sun{
private:
    vec3f m_position;
    float m_height;
    int m_sunshape;
    vec3f m_sundir;
    bool is_device_memory;
public:
    __host__ Sun();
    __host__ Sun(const vec3f& postion,float height,int sunshape);
    __host__ ~Sun();

    Sun(const Sun&) = delete;
    Sun& operator=(const Sun&) = delete;

    // 内存管理
    __host__ Sun* CreateDeviceCopy();

    __host__ void ConvertToDirection(float heightAngle, float azimuthAngle);

    __device__ vec3f SampleGaussianSunshape(curandState* local_state,float gauss_sigma); 
    __device__ vec3f SampleUniformDiskSunshape(curandState* local_state,float solar_radius_rad);
    __device__ vec3f SampleBuieSunshape(curandState* local_state,float csr);

    __host__ __device__ vec3f GetDirection() const {return m_sundir;};
    __host__ __device__ vec3f GetPosition() const {return m_position;};
    __host__ int GetSunshape() const {return m_sunshape;};
    __host__ __device__ void SetPosition(const vec3f& position) {m_position = position;};
    __host__ __device__ void SetHeight(float height) {m_height = height;};
    __host__ __device__ void SetSunshape(int sunshape) {m_sunshape = sunshape;};
    __host__ __device__ void SetSunDir(const vec3f& sundir) {m_sundir = sundir;};

    __host__ void display_DEBUG_info();
};