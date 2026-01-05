#pragma once
#include <fstream>
#include <sstream>
#include "SF_mcrt_CPC.cuh"
#include "vector.cuh"

struct pixel_distribution_on_plane{
    int pixel_x,pixel_y;
    int pixelID;
    vec3f mu1,mu2;
    float w1,w2,kappa1,kappa2;
};

void load_distribution(std::vector<pixel_distribution_on_plane>& h_pixels, const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    std::string line;
    int index = 0;
    
    // 跳过标题行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string token;
        
        // 读取pixel_x, pixel_y
        std::getline(ss, token, ',');
        int pixel_x = std::stoi(token);
        std::getline(ss, token, ',');
        int pixel_y = std::stoi(token);
        
        // 计算pixelID
        h_pixels[index].pixel_x = pixel_x;
        h_pixels[index].pixel_y = pixel_y;
        h_pixels[index].pixelID = pixel_y * plane_Wpix + pixel_x;
        
        // 跳过rays_count1, rays_count2
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');
        
        // 读取mu1 (avg_dir1_x, avg_dir1_y, avg_dir1_z)
        std::getline(ss, token, ',');
        h_pixels[index].mu1.x = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].mu1.y = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].mu1.z = std::stof(token);
        
        // 读取mu2 (avg_dir2_x, avg_dir2_y, avg_dir2_z)
        std::getline(ss, token, ',');
        h_pixels[index].mu2.x = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].mu2.y = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].mu2.z = std::stof(token);
        
        // 跳过avg_dir_length1, avg_dir_length2
        std::getline(ss, token, ',');
        std::getline(ss, token, ',');
        
        // 读取w1, w2, kappa1, kappa2
        std::getline(ss, token, ',');
        h_pixels[index].w1 = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].w2 = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].kappa1 = std::stof(token);
        std::getline(ss, token, ',');
        h_pixels[index].kappa2 = std::stof(token);
        
        index++;
    }
    
    file.close();
}

/*
    @brief 由一次镜几何分布设置虚拟平面上射线出射方向
    @param origin 射线原点
    @param refID 反射镜ID
    @param Reflectors_ptr 反射镜数组
    @param local_state 本地随机数状态
    @return vec3f 出射射线方向
*/
// __device__ vec3f set_direction(const vec3f& origin,int refID,reflector* Reflectors_ptr,curandState *local_state){
//     vec3f direction = Reflectors_ptr[refID].aim;
//     vec3f A=add3(Reflectors_ptr[refID].position,{-reflector_width/2,-reflector_len/2,0.0f});
//     vec3f B=add3(Reflectors_ptr[refID].position,{-reflector_width/2,reflector_len/2,0.0f});
//     vec3f C=add3(Reflectors_ptr[refID].position,{reflector_width/2,-reflector_len/2,0.0f});
//     vec3f D=add3(Reflectors_ptr[refID].position,{reflector_width/2,reflector_len/2,0.0f});
//     A=rotate3(A,Reflectors_ptr[refID].normal);
//     B=rotate3(B,Reflectors_ptr[refID].normal);
//     C=rotate3(C,Reflectors_ptr[refID].normal);
//     D=rotate3(D,Reflectors_ptr[refID].normal);//BA=OA-OB
//     vec3f AO=sub3(origin,A);
//     vec3f BO=sub3(origin,B);
//     vec3f CO=sub3(origin,C);
//     vec3f DO=sub3(origin,D);
//     float theta_min = fminf(fminf(atan2f(AO.z,AO.y),PI-atan2f(BO.z,BO.y)),fminf(atan2f(CO.z,CO.y),PI-atan2f(DO.z,DO.y)));
//     float theta = theta_min+curand_uniform(local_state)*(PI-2.0f*theta_min);
//     if(theta<PI/2.0f)
//         direction.y = sinf(theta)*sqrtf(direction.x*direction.x+direction.z*direction.z);
//     else
//         direction.y = -sinf(theta)*sqrtf(direction.x*direction.x+direction.z*direction.z);
//     direction = normalize3(direction);
//     if(0)
//         printf("origin=(%.2f,%.2f,%.2f),ref[%d]_pos=(%.2f,%.2f,%.2f),theta_A=%.2f,theta_B=%.2f,theta_C=%.2f,theta_D=%.2f,theta_min=%.2f,theta=%.2f,,dir=(%.2f,%.2f,%.2f)\n",
//             origin.x,origin.y,origin.z,refID,Reflectors_ptr[refID].position.x,Reflectors_ptr[refID].position.y,Reflectors_ptr[refID].position.z,
//             atan2f(AO.z,AO.y)*180.0f/PI,(PI-atan2f(BO.z,BO.y))*180.0f/PI,atan2f(CO.z,CO.y)*180.0f/PI,(PI-atan2f(DO.z,DO.y))*180.0f/PI,theta_min*180.0f/PI, theta*180.0f/PI,direction.x,direction.y,direction.z);
//     return direction;
// }

__device__ vec3f generate_dir_on_apperture(const vec3f &target_dir,curandState *local_state){
    float ra3 = curand_uniform(local_state);
    float phi_ray = 2.0f * (float)PI * ra3;
    float theta_ray = asinf(sqrtf(curand_uniform(local_state)));
    vec3f direction = {
        cosf(phi_ray) * sinf(theta_ray),
        sinf(phi_ray) * sinf(theta_ray),
        cosf(theta_ray)};
    return direction;
}

__device__ vec3f sample_vMF(const vec3f& mu, const float& kappa, curandState* local_state)
{
    const int dim = 3;
    if(kappa < 1e-6f){
        // 使用球面均匀分布采样
        float phi = 2.0f * PI * curand_uniform(local_state);
        float z = 2.0f * curand_uniform(local_state) - 1.0f;
        float r = sqrtf(1.0f - z * z);
        return vec3f{r * cosf(phi), r * sinf(phi), z};
    }

    // 使用Ulrich算法采样vMF分布
    float b = (-2.0f * kappa + sqrtf(4.0f * kappa * kappa + (dim - 1) * (dim - 1))) / (dim - 1);
    float x0 = (1.0f - b) / (1.0f + b);
    float c = kappa * x0 + (dim - 1) * logf(1.0f - x0 * x0);

    float w = 1.0f;
    float x = 0.0f;
    while(true){
        float z = curand_uniform(local_state);
        float beta = (1.0f - x0) / (1.0f + x0);
        x = (1.0f - (1.0f + beta) * z) / (1.0f - (1.0f - beta) * z);
        
        // 计算接受概率
        w = kappa * x + (dim - 1) * logf(1.0f - x0 * x) - c;
        
        // 接受检查
        if(w >= logf(curand_uniform(local_state))){
            break;
        }
    }
    float xi1 = curand_uniform(local_state);
    float xi2 = curand_uniform(local_state);
    
    float phi = 2.0f * PI * xi2;
    float sin_theta = sqrtf(1.0f - x * x);
    
    // 局部坐标系中的方向向量
    vec3f v_local = {
        sin_theta * cosf(phi),
        sin_theta * sinf(phi),
        x
    };
    
    // 旋转到目标方向 mu
    // 构建正交基
    vec3f v1, v2;
    
    // 选择与 mu 不平行的任意向量
    vec3f t = fabsf(mu.x) > 0.1f ? vec3f{0.0f, 1.0f, 0.0f} : vec3f{1.0f, 0.0f, 0.0f};
    
    // Gram-Schmidt 正交化
    v1 = normalize3(cross3(mu, t));
    v2 = normalize3(cross3(mu, v1));
    
    // 将局部坐标旋转到 mu 方向
    vec3f result = {
        v_local.x * v1.x + v_local.y * v2.x + v_local.z * mu.x,
        v_local.x * v1.y + v_local.y * v2.y + v_local.z * mu.y,
        v_local.x * v1.z + v_local.y * v2.z + v_local.z * mu.z
    };
    
    return normalize3(result);
}

__device__ vec3f sample_vMF_mixture_K2(const pixel_distribution_on_plane& d_pixel,curandState *local_state)
{
    float w1 = d_pixel.w1,w2 = d_pixel.w2;
    float kappa1 = d_pixel.kappa1,kappa2=d_pixel.kappa2;
    vec3f mu1 = d_pixel.mu1,mu2=d_pixel.mu2;
    float p = curand_uniform(local_state);
    if(p<w1)
        return sample_vMF(mu1,kappa1,local_state);
    else
        return sample_vMF(mu2,kappa2,local_state);
}