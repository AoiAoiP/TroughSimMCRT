#pragma once
#include <cmath>
#include <vector>
#include "SF_mcrt_CPC.cuh"
#include "vector.cuh"

struct reflector {
    vec3f position;
    vec3f normal = {0.0f, 0.0f, 1.0f};
    vec3f aim = {0.0f, 0.0f, receiver_h};//设定预期瞄准位置
    float length = reflector_len;
    float width = reflector_width;
};

void set_position(std::vector<reflector>& Reflector){
    float o = - (reflector_num-1) * 0.5 * (reflector_width + intv);
    for(int i=0;i<(reflector_num+1)/2;i++){
        float xi = o + i*(intv+reflector_width);
        Reflector[i].position = vec3f{xi, 0.0f, 0.0f};
        Reflector[reflector_num-1-i].position = vec3f{-xi, 0.0f, 0.0f};
    }
}


/*
    @brief host：根据太阳入射方向计算所有反射镜的法向
    @param sun_dir 太阳入射方向
    @param Reflectors 反射镜数组
*/
void compute_normal(vec3f sun_dir,
    std::vector<reflector>& Reflectors){
        // ？懒狗为什么不设计host端的向量各计算？
        for(int i=0;i<reflector_num;i++){
            // 入射方向（从太阳到反射镜）
            vec3f incident_dir = {-sun_dir.x, -sun_dir.y, -sun_dir.z};
            float inc_len = sqrt(incident_dir.x*incident_dir.x + incident_dir.y*incident_dir.y + incident_dir.z*incident_dir.z);
            if (inc_len > 1e-12) {
                incident_dir.x /= inc_len;
                incident_dir.y /= inc_len;
                incident_dir.z /= inc_len;
            }
            
            // 目标方向（从反射镜到接收平面中心）
            vec3f target_dir = {Reflectors[i].aim.x - Reflectors[i].position.x,
                                Reflectors[i].aim.y - Reflectors[i].position.y,
                                Reflectors[i].aim.z - Reflectors[i].position.z};
            float target_len = sqrt(target_dir.x*target_dir.x + target_dir.y*target_dir.y + target_dir.z*target_dir.z);
            if (target_len > 1e-12) {
                target_dir.x /= target_len;
                target_dir.y /= target_len;
                target_dir.z /= target_len;
            }
            
        
            // 计算法向：入射方向和目标方向的角平分线
            float nx = incident_dir.x + target_dir.x;
            float ny = incident_dir.y + target_dir.y;
            float nz = incident_dir.z + target_dir.z;
            // 归一化法向
            float n_len = sqrt(nx*nx + ny*ny + nz*nz);
            if (n_len > 1e-12) {
                nx /= n_len;
                ny /= n_len;
                nz /= n_len;
            }
            // printf("refID=%d,position=(%.4f,%.4f,%.4f),incident={%.4f,%.4f,%.4f},target={%.4f,%.4f,%.4f},normal={%.4f,%.4f,%.4f}\n",
            // i,Reflectors[i].position.x,Reflectors[i].position.y,Reflectors[i].position.z,
            // incident_dir.x,incident_dir.y,incident_dir.z,
            // target_dir.x,target_dir.y,target_dir.z,
            // nx,ny,nz);
            // 确保法向朝上
            if (nz < 0) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            Reflectors[i].normal = {nx,ny,nz};
        }
}

/*
    @brief host：根据太阳入射方向计算所有反射镜的法向
    @param sun_dir 太阳入射方向
    @param Reflectors 反射镜数组
*/
void compute_aim(std::vector<reflector>& Reflectors){
        for(int i=0;i<reflector_num;i++){
            // 目标方向（从反射镜到接收平面中心）
            Reflectors[i].aim = {0.0f - Reflectors[i].position.x,
                                0.0f - Reflectors[i].position.y,
                                receiver_h - Reflectors[i].position.z};
            float target_len = sqrt(Reflectors[i].aim.x*Reflectors[i].aim.x + Reflectors[i].aim.y*Reflectors[i].aim.y + Reflectors[i].aim.z*Reflectors[i].aim.z);
            if (target_len > 1e-12) {
                Reflectors[i].aim={Reflectors[i].aim.x/target_len, Reflectors[i].aim.y/target_len, Reflectors[i].aim.z/target_len};
            }
        }
}

void init_reflector(const vec3f& sun_dir,std::vector<reflector>& Reflectors){
    set_position(Reflectors);
    compute_normal(sun_dir, Reflectors);
    compute_aim(Reflectors);
}