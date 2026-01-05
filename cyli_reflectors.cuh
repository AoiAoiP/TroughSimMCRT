#pragma once
#include "SF_mcrt_CPC.cuh"
#include "vector.cuh"

struct CyliReflector {
    vec3f position;
    vec3f normal;
    vec3f aim; //设定预期瞄准位置
    vec3f nor; // 圆心位置
    float length = reflector_len;
    float width = reflector_width;
    float R;
    int l_pixel = Lpixel;
    int w_pixel = Wpixel;
    float slope = slope_error[1];

    __host__ __device__ CyliReflector();
    __host__ __device__ CyliReflector(const vec3f& postion,const vec3f& normal,const vec3f& aim,const vec3f& nor,float length,float width,int l_pixel,int w_pixel,float slope);
};

class CyliReflectors{
private:
    CyliReflector* p_reflectors;
    int num_CyliReflectors;
    bool is_device_memory;
public:
    // 构造函数
    __host__  CyliReflectors();
    __host__  CyliReflectors(int num,const vec3f& sun_dir);
    __host__  ~CyliReflectors();
    __host__ __device__ CyliReflector* getdReflectors();

    CyliReflectors(const CyliReflectors&) = delete;
    CyliReflectors& operator=(const CyliReflectors&) = delete;

    // 内存管理
    __host__ void allocateDeviceMemory();
    __host__ void freeDeviceMemory();
    __host__ void cpyHostToDevice();
    __host__ void cpyDeviceToHost();
    __host__ CyliReflectors* createDeviceCopy();

    // 核心功能
    __host__ void setPosition(); // 设置位置
    __host__ __device__ vec3f getRefPosition(int RefID); // 获取RefID号镜子的位置
    __host__ void calculateCurvatureRadii(); // 计算曲率半径
    __host__ void calculateAndsetNor(const vec3f& sun_dir); // 计算并设置圆心位置（局部坐标）
    __host__ __device__ vec3f getRefNormalatPoint(int RefID,const vec3f& point); // 获取RefID号镜点point处的法向（局部坐标）
    __host__ __device__ vec3f getRefNormal(int RefID); // 获取RefID号镜子上的法向
    __device__ vec3f sample_on_CyliRefs(int RefID,int pixelID,curandState* rand_state); // 采样在镜子上的点
    __host__ void display_DEBUG_info();
};
