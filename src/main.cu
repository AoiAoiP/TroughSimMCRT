// main.cu
#include "app.cuh"

__global__ void render(float* d_flux_map,int* d_hit_count){
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_sim_config.total_rays) return;

    float f = d_trough_config.focal_length;
    float2 rand_origin = Optics::get_random_pair(idx,DIM_ORIGIN_GEN);
    int mirror_index;
    float3 ray_origin = Geometry::sampleTrough(rand_origin.x,rand_origin.y,d_trough_config,mirror_index);
    float3 normal_ideal = normalize(make_float3(-ray_origin.x/(2.0f*f),0.0f,1.0f));

    float2 rand_sun = Optics::get_random_pair(idx,DIM_SUN_SHAPE);
    float3 ray_in = normalize(Optics::SampleSunshape(rand_sun.x,rand_sun.y,d_sun_config));
    ray_in = -ray_in;

    // float cos_theta = dot(ray_in,normal_ideal);
    // float ray_weight = -cos_theta;

    float3 ray_ref_ideal = reflect(ray_in,normal_ideal);
    
    float sigma_total = sqrtf(4*d_trough_config.slope_error*d_trough_config.slope_error+d_trough_config.specularity_error*d_trough_config.specularity_error)*0.001f;
    float2 rand_pert = Optics::get_random_pair(idx,DIM_PERT_ERR);
    float3 ray_ref = normalize(Optics::GaussianPerturb(ray_ref_ideal,sigma_total,rand_pert.x,rand_pert.y));
    
    Geometry::HitInfo hit = Geometry::intersectAbsorber(ray_origin,ray_ref,d_absorber_config);
    if(hit.is_hit){
        atomicAdd(d_hit_count,1);      
        float hit_angle = atan2f(hit.hit_point.x- d_absorber_config.position.x, hit.hit_point.z - d_absorber_config.position.z);
        if (hit_angle < 0.0f) hit_angle += 2.0f * PI;
        int bin_x = (int)((hit_angle / (2.0f * PI)) * d_sim_config.grid_res_x);
        int bin_z = (int)(((hit.hit_point.y - d_absorber_config.position.y+ d_absorber_config.length * 0.5f) / d_absorber_config.length) * d_sim_config.grid_res_z);

        // 边界保护
        bin_x = min(max(bin_x, 0), d_sim_config.grid_res_x - 1);
        bin_z = min(max(bin_z, 0), d_sim_config.grid_res_z - 1);

        int grid_index = bin_z * d_sim_config.grid_res_x + bin_x;
#ifdef DEBUG
        if(idx < 1e2)
            printf("idx=%ld,ray_origin=(%.4f,%.4f,%.4f),normal_ideal=(%.4f,%.4f,%.4f),ray_in=(%.4f,%.4f,%.4f),ray_ref=(%.4f,%.4f,%.4f),hit_point=(%.4f,%.4f,%.4f)\n",
                idx,ray_origin.x,ray_origin.y,ray_origin.z,normal_ideal.x,normal_ideal.y,normal_ideal.z,ray_in.x,ray_in.y,ray_in.z,ray_ref_ideal.x,ray_ref_ideal.y,ray_ref_ideal.z,hit.hit_point.x,hit.hit_point.y,hit.hit_point.z);
#endif   
        atomicAdd(&d_flux_map[grid_index],  d_trough_config.reflectivity);
    }
}

int main(){
    loadTroughConfigToGPU("../resources/config.json");
    loadSunConfigToGPU("../resources/config.json");
    loadAbsorberConfigToGPU("../resources/config.json");

    SimConfig h_sim_config = loadSimConfigToGPU("../resources/config.json");

    int N = h_sim_config.total_rays;
    int blockSize = h_sim_config.block_size;
    int gridSize = (N+blockSize-1)/blockSize;

    int grid_size = h_sim_config.grid_res_z * h_sim_config.grid_res_x;
    size_t flux_map_size = grid_size * sizeof(float);
    float* h_flux_map = (float*)malloc(flux_map_size);
    float* d_flux_map = nullptr;
    cudaMalloc((void**)&d_flux_map, flux_map_size);
    cudaMemset(d_flux_map, 0, flux_map_size);
    int h_hit_count = 0;
    int* d_hit_count = nullptr;
    cudaMalloc((void**)&d_hit_count,sizeof(int));
    cudaMemset(d_hit_count,0,sizeof(int));
    
    Optics::initRandomPools(Optics::POOL_SIZE,N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout << "Launching Kernel with Grid: " << gridSize << ", Block: " << blockSize <<", N:"<<N<<std::endl;
    cudaEventRecord(start);
    render<<<gridSize, blockSize>>>(d_flux_map,d_hit_count);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if(err!=cudaSuccess){
        std::cerr << "Kernel Execution Error: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_flux_map,d_flux_map,flux_map_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_hit_count,d_hit_count,sizeof(int),cudaMemcpyDeviceToHost);
    float total_intercepted_energy = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        total_intercepted_energy += h_flux_map[i];
    }
    float intercept_factor = (float)h_hit_count / (float)N;
    float optical_efficiency = total_intercepted_energy / (float)N;
    std::cout << "------------------------------------------\n";
    std::cout << "Physical Hits: " << h_hit_count << " / " << N << std::endl;
    std::cout << "Intercept Factor (Geometric): " << intercept_factor * 100.0f << " %" << std::endl;
    std::cout << "Optical Efficiency (w/ Reflectivity): " << optical_efficiency * 100.0f << " %" << std::endl;
    std::cout << "------------------------------------------\n";

    std::ofstream outfile("flux_map.csv");
    if (outfile.is_open()) {
        for (int z = 0; z < h_sim_config.grid_res_z; ++z) {
            for (int x = 0; x < h_sim_config.grid_res_x; ++x) {
                // 一维索引转二维坐标
                int index = z * h_sim_config.grid_res_x + x;
                outfile << h_flux_map[index];
                if (x != h_sim_config.grid_res_x - 1) {
                    outfile << ",";
                }
            }
            outfile << "\n"; // 换行
        }
        outfile.close();
        std::cout << "Flux map successfully saved to 'flux_map.csv'." << std::endl;
    } else {
        std::cerr << "Failed to open flux_map.csv for writing!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    Optics::freeRandomPools();
    cudaFree(d_flux_map);
    cudaFree(d_hit_count);
    free(h_flux_map);
    return 0;
}