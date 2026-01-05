#include "cyli_reflectors.cuh"

__host__ __device__  CyliReflector::CyliReflector()
    : position{0.0f,0.0f,0.0f},normal{0.0f,0.0f,1.0f},aim{0.0f,0.0f,receiver_h},nor{0.0f,0.0f,1.0f},
      length{reflector_len},width{reflector_width},R{0.0f},l_pixel{Lpixel},w_pixel{Wpixel},slope{slope_error[1]}{}
__host__ __device__  CyliReflector::CyliReflector(const vec3f& postion,const vec3f& normal,const vec3f& aim,const vec3f& nor,float length,float width,int l_pixel,int w_pixel,float slope)
    : position{postion},normal{normal},aim{aim},nor{nor},length{length},R{R},width{width},slope{slope}{}

__host__ CyliReflectors::CyliReflectors()
    : p_reflectors{nullptr},num_CyliReflectors{0},is_device_memory{false}{}

__host__ CyliReflectors::CyliReflectors(int num,const vec3f& sun_dir)
    : p_reflectors{nullptr},num_CyliReflectors{num},is_device_memory{false}{
        p_reflectors = new CyliReflector[num];
        for (int i = 0; i < num; i++) {
            p_reflectors[i] = CyliReflector();  // 调用默认构造函数
        }
        setPosition();
        calculateCurvatureRadii();
        calculateAndsetNor(sun_dir);
    }

__host__ CyliReflectors::~CyliReflectors()
{
    if (!is_device_memory && p_reflectors != nullptr) {
        delete[] p_reflectors;
    }
}

__host__ void CyliReflectors::allocateDeviceMemory() {
    if (p_reflectors == nullptr || num_CyliReflectors == 0) return;
    
    CyliReflector* d_ptr;
    CHECK(cudaMalloc(&d_ptr, num_CyliReflectors * sizeof(CyliReflector)));
    CHECK(cudaMemcpy(d_ptr, p_reflectors, num_CyliReflectors * sizeof(CyliReflector), 
                     cudaMemcpyHostToDevice));
    
    delete[] p_reflectors; // 删除主机内存
    p_reflectors = d_ptr;
    is_device_memory = true;
}

__host__ void CyliReflectors::freeDeviceMemory() {
    if (is_device_memory && p_reflectors != nullptr) {
        CHECK(cudaFree(p_reflectors));
        p_reflectors = nullptr;
        is_device_memory = false;
    }
}

__host__ void CyliReflectors::cpyHostToDevice() {
    if (p_reflectors != nullptr && num_CyliReflectors > 0 && !is_device_memory) {
        CyliReflector* d_ptr;
        CHECK(cudaMalloc(&d_ptr, num_CyliReflectors * sizeof(CyliReflector)));
        CHECK(cudaMemcpy(d_ptr, p_reflectors, num_CyliReflectors * sizeof(CyliReflector), 
                         cudaMemcpyHostToDevice));
        
        delete[] p_reflectors;
        p_reflectors = d_ptr;
        is_device_memory = true;
    }
}

__host__ void CyliReflectors::cpyDeviceToHost() {
    if (p_reflectors != nullptr && num_CyliReflectors > 0 && is_device_memory) {
        CyliReflector* h_ptr = new CyliReflector[num_CyliReflectors];
        CHECK(cudaMemcpy(h_ptr, p_reflectors, num_CyliReflectors * sizeof(CyliReflector), 
                         cudaMemcpyDeviceToHost));
        
        CHECK(cudaFree(p_reflectors));
        p_reflectors = h_ptr;
        is_device_memory = false;
    }
}

__host__ CyliReflectors* CyliReflectors::createDeviceCopy() {
    CyliReflectors* d_ptr;
    CHECK(cudaMalloc(&d_ptr, sizeof(CyliReflectors)));
    
    // 创建设备端的反射器数组
    CyliReflector* d_reflectors;
    CHECK(cudaMalloc(&d_reflectors, num_CyliReflectors * sizeof(CyliReflector)));
    CHECK(cudaMemcpy(d_reflectors, p_reflectors, num_CyliReflectors * sizeof(CyliReflector), 
                     cudaMemcpyHostToDevice));
    
    // 创建临时主机对象用于复制到设备
    CyliReflectors temp_host;  // 使用默认构造函数
    temp_host.p_reflectors = d_reflectors;
    temp_host.num_CyliReflectors = num_CyliReflectors;
    temp_host.is_device_memory = true;
    
    CHECK(cudaMemcpy(d_ptr, &temp_host, sizeof(CyliReflectors), cudaMemcpyHostToDevice));
    return d_ptr;
}

__host__ __device__ CyliReflector* CyliReflectors::getdReflectors(){
    return this->p_reflectors;
}

__host__ void CyliReflectors::setPosition(){
    float o = - (num_CyliReflectors-1) * 0.5 * (p_reflectors[0].width + intv);
    for(int i=0;i<(num_CyliReflectors+1)/2;i++){
        float xi = o + i*(intv+p_reflectors[0].width);
        p_reflectors[i].position = vec3f{xi, 0.0f, 0.0f};
        p_reflectors[num_CyliReflectors-1-i].position = vec3f{-xi, 0.0f, 0.0f};
    }
}

__host__ __device__ vec3f CyliReflectors::getRefPosition(int RefID){
    return p_reflectors[RefID].position;
}

__host__ void CyliReflectors::calculateCurvatureRadii(){
    for(int i=0;i<num_CyliReflectors;i++){
        p_reflectors[i].R = 2.0f * sqrtf(p_reflectors[i].position.x*p_reflectors[i].position.x + p_reflectors[i].aim.z*p_reflectors[i].aim.z);
    }
}

__host__ void CyliReflectors::calculateAndsetNor(const vec3f& sun_dir){
    for(int i=0;i<num_CyliReflectors;i++){
        vec3f refs=normalize3(sub3(p_reflectors[i].aim,p_reflectors[i].position));
        vec3f sun=normalize3(mul3(sun_dir,-1.0f));
        vec3f nor=normalize3({(refs.x+sun.x)/2.0f,(refs.y+sun.y)/2.0f,(refs.z+sun.z)/2.0f});
        // printf("idir = (%.4f,%.4f,%.4f), rdir = (%.4f,%.4f,%.4f), set_nor = (%.4f,%.4f,%.4f)\n",sun.x,sun.y,sun.z,refs.x,refs.y,refs.z,nor.x,nor.y,nor.z);
        p_reflectors[i].nor = nor;
    }
}

__host__ __device__ vec3f CyliReflectors::getRefNormalatPoint(int RefID,const vec3f& point){
    vec3f circle_center = {0.0f,point.y,p_reflectors[RefID].R};
    return normalize3(sub3(circle_center,point));
}

__host__ __device__ vec3f CyliReflectors::getRefNormal(int RefID){
    return p_reflectors[RefID].nor;
}

__device__ vec3f CyliReflectors::sample_on_CyliRefs(int RefID,int pixelID,curandState* rand_state){
    int col = pixelID % p_reflectors[RefID].w_pixel; // 获取x轴坐标,即width方向坐标,w_pixel=60
    int row = pixelID / p_reflectors[RefID].w_pixel; // 获取y轴坐标

    // 生成y偏移并应用
    float base_y = (row + 0.5f) * (p_reflectors[RefID].length / p_reflectors[RefID].l_pixel) - p_reflectors[RefID].length * 0.5f;
    float offset_y = (curand_uniform(rand_state) - 0.5f) * (p_reflectors[RefID].length / p_reflectors[RefID].l_pixel);
    float coory = base_y + offset_y;

    // 生成theta偏移并应用到xz
    float R = p_reflectors[RefID].R;
    float arclength = p_reflectors[RefID].width;
    float u = curand_uniform(rand_state);
    float total_theta = 2.0f * asinf((arclength*0.5f)/R);
    float dtheta = total_theta / p_reflectors[RefID].w_pixel;
    float theta = -total_theta*0.5f + (col + u)*dtheta;
    float coorx = R * sinf(theta);
    float coorz = R * cosf(theta) - R;
    // if (col == 23)
        // printf("(col,row)=(%d,%d),theta=%.6f,coor={%.6f,%.6f,%.6f}\n", col, row,theta,coorx, coory, coorz);
    return vec3f{coorx, coory, coorz};
}

__host__ void CyliReflectors::display_DEBUG_info(){
        for(int i=0;i<num_CyliReflectors;i++){
            printf("reflector %d: position = (%.4f, %.4f, %.4f), aim = (%.4f, %.4f, %.4f), nor(局坐标) = (%.4f, %.4f, %.4f),R = %.4f, C=%.6f\n",
                   i, p_reflectors[i].position.x, p_reflectors[i].position.y, p_reflectors[i].position.z,
                   p_reflectors[i].aim.x, p_reflectors[i].aim.y, p_reflectors[i].aim.z,
                   p_reflectors[i].nor.x, p_reflectors[i].nor.y, p_reflectors[i].nor.z,
                   p_reflectors[i].R, 1.0f/p_reflectors[i].R);
        }
}