#include "config/sim_config.cuh"

__constant__ SimConfig d_sim_config;

using json = nlohmann::json;

SimConfig loadSimConfigToGPU(const std::string& filepath){
    std::ifstream file(filepath);
    if(!file.is_open()) {
        std::cerr << "Failed to open config file: " << filepath << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    SimConfig h_sim_config;

    auto sim_json = j["Simulation"];
    h_sim_config.total_rays = sim_json["total_rays"].get<int>();
    h_sim_config.block_size = sim_json["block_size"].get<int>();
    h_sim_config.grid_res_z = sim_json["grid_res_z"].get<int>();
    h_sim_config.grid_res_x = sim_json["grid_res_x"].get<int>();

    cudaError_t err = cudaMemcpyToSymbol(d_sim_config, &h_sim_config, sizeof(SimConfig));
    if(err != cudaSuccess) {
        std::cerr << "Failed to copy config to GPU symbol: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "Successfully loaded sim config to GPU.\n";

    return h_sim_config;
}