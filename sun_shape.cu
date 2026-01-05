#include "sun_shape.cuh"

__host__ Sun::Sun()
    :m_position({0.0f,0.0f,0.0f}),m_height(0.0f),m_sunshape(0),m_sundir({0.0f,0.0f,-1.0f}),is_device_memory(false){}

__host__ Sun::Sun(const vec3f& postion,float height,int sunshape)
    : m_position{postion},m_height{height},m_sunshape{sunshape},m_sundir({0.0f,0.0f,-1.0f}),is_device_memory(false) {}

__host__ Sun::~Sun() {}


__host__ Sun* Sun::CreateDeviceCopy() {
    Sun* d_ptr;
    CHECK(cudaMalloc(&d_ptr, sizeof(Sun)));
    
    // 创建临时主机对象用于复制到设备
    Sun temp_host;
    temp_host.m_position = m_position;
    temp_host.m_height = m_height;
    temp_host.m_sunshape = m_sunshape;
    temp_host.m_sundir = m_sundir;
    temp_host.is_device_memory = true;
    
    CHECK(cudaMemcpy(d_ptr, &temp_host, sizeof(Sun), cudaMemcpyHostToDevice));
    return d_ptr;
}

__host__ void Sun::ConvertToDirection(float heightAngle, float azimuthAngle){
    // 将角度转换为弧度
    float elev_rad = heightAngle * PI / 180.0f;
    float azim_rad = azimuthAngle * PI / 180.0f;
    
    // 正确的球坐标转直角坐标公式
    float sunx = cos(elev_rad) * sin(azim_rad);
    float suny = cos(elev_rad) * cos(azim_rad);
    float sunz = -sin(elev_rad);  // 负号因为太阳在天上，光线向下
    
    m_position = vec3f{sunx, suny, sunz};
    m_sundir = normalize3(m_position);
}

/*
    @brief 高斯太阳采样
    @param local_state 随机数状态
    @param solar_radius_rad 太阳半径（弧度）
    @return 太阳光线方向
*/
__device__ vec3f Sun::SampleGaussianSunshape(curandState* local_state,float gauss_sigma) {
    float dx_ang = curand_normal(local_state) * gauss_sigma;
    float dy_ang = curand_normal(local_state) * gauss_sigma;

    // 建立局部坐标系
    vec3f s = normalize3(m_sundir);
    vec3f arbitrary_up = {0.0f, 1.0f, 0.0f};
    vec3f uvec = cross3(s, arbitrary_up);
    float ulen = dot3(uvec, uvec);
    
    if (ulen < EPSILON) { 
        arbitrary_up = {1.0f, 0.0f, 0.0f};
        uvec = cross3(s, arbitrary_up);
        ulen = dot3(uvec, uvec);
        if (ulen < EPSILON) {
            return m_sundir; // 无法建立坐标系，返回原方向
        }
    }

    uvec = normalize3(uvec);
    vec3f vvec = cross3(uvec, s);
    
    // 应用太阳形状偏移
    vec3f sampled_dir = {
        m_sundir.x + dx_ang * uvec.x + dy_ang * vvec.x,
        m_sundir.y + dx_ang * uvec.y + dy_ang * vvec.y,
        m_sundir.z + dx_ang * uvec.z + dy_ang * vvec.z
    };
    
    return normalize3(sampled_dir);
}

/*
    @brief 均匀圆盘太阳采样
    @param local_state 随机数状态
    @param solar_radius_rad 太阳半径（弧度）
    @return 太阳光线方向
*/
__device__ vec3f Sun::SampleUniformDiskSunshape(curandState* local_state,float solar_radius_rad) {
    float r = solar_radius_rad * sqrtf(curand_uniform(local_state));
    float phi = 2.0f * PI * curand_uniform(local_state);
    float dx_ang = r * cosf(phi);
    float dy_ang = r * sinf(phi);

    // 建立局部坐标系
    vec3f s = normalize3(m_sundir);
    vec3f arbitrary_up = {0.0f, 1.0f, 0.0f};
    vec3f uvec = cross3(s, arbitrary_up);
    float ulen = dot3(uvec, uvec);
    
    if (ulen < EPSILON) { 
        arbitrary_up = {1.0f, 0.0f, 0.0f};
        uvec = cross3(s, arbitrary_up);
        ulen = dot3(uvec, uvec);
        if (ulen < EPSILON) {
            return m_sundir; // 无法建立坐标系，返回原方向
        }
    }

    uvec = normalize3(uvec);
    vec3f vvec = cross3(uvec, s);
    
    // 应用太阳形状偏移
    vec3f sampled_dir = {
        m_sundir.x + dx_ang * uvec.x + dy_ang * vvec.x,
        m_sundir.y + dx_ang * uvec.y + dy_ang * vvec.y,
        m_sundir.z + dx_ang * uvec.z + dy_ang * vvec.z
    };
    
    return normalize3(sampled_dir);
}

/*
    @brief Buie太阳采样
    @param local_state 随机数状态
    @param csr 太阳半径（弧度）
    @return 太阳光线方向
*/
__device__ vec3f Sun::SampleBuieSunshape(curandState* local_state,float csr) {
    const float theta_disk = 4.65e-3f;
    const float theta_aureole = 9.3e-3f;
    
    // 限制csr范围
    csr = fmaxf(csr, 0.05f);
    csr = fminf(csr, 0.5f);
    
    float theta_buie;
    float rand_val = curand_uniform(local_state);
    
    // 根据面积比例选择区域，避免拒绝采样
    if (rand_val < 0.98f) { // 约98%的光线在太阳盘内
        // 太阳盘：均匀采样
        theta_buie = curand_uniform(local_state) * theta_disk;
    } else {
        // 光晕：使用近似的高斯分布
        float mean_theta = (theta_disk + theta_aureole) * 0.5f;
        float sigma = (theta_aureole - theta_disk) * 0.5f;
        theta_buie = fabsf(curand_normal(local_state)) * sigma + mean_theta;
        theta_buie = fminf(theta_buie, theta_aureole);
    }
    
    // 随机选择正负角度
    if (curand_uniform(local_state) < 0.5f) {
        theta_buie = -theta_buie;
    }
    
    // 转换为方向向量
    float phi = 2.0f * PI * curand_uniform(local_state);
    float dx_ang = sinf(theta_buie) * cosf(phi);
    float dy_ang = sinf(theta_buie) * sinf(phi);

    // 建立局部坐标系
    vec3f s = normalize3(m_sundir);
    vec3f arbitrary_up = {0.0f, 1.0f, 0.0f};
    vec3f uvec = cross3(s, arbitrary_up);
    float ulen = dot3(uvec, uvec);
    
    if (ulen < EPSILON) { 
        arbitrary_up = {1.0f, 0.0f, 0.0f};
        uvec = cross3(s, arbitrary_up);
        ulen = dot3(uvec, uvec);
        if (ulen < EPSILON) {
            return m_sundir; // 无法建立坐标系，返回原方向
        }
    }

    uvec = normalize3(uvec);
    vec3f vvec = cross3(uvec, s);
    
    // 应用太阳形状偏移
    vec3f sampled_dir = {
        m_sundir.x + dx_ang * uvec.x + dy_ang * vvec.x,
        m_sundir.y + dx_ang * uvec.y + dy_ang * vvec.y,
        m_sundir.z + dx_ang * uvec.z + dy_ang * vvec.z
    };
    
    return normalize3(sampled_dir);
}

__host__ void Sun::display_DEBUG_info(){
    printf("Sun_pos = (%.4f, %.4f, %.4f), sun_height=%.2f,sun_dir = (%.4f, %.4f, %.4f)\n",
            m_position.x, m_position.y, m_position.z,
            m_height,m_sundir.x, m_sundir.y, m_sundir.z);
}