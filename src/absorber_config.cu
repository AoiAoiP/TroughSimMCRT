#include "config/absorber_config.cuh"

using json = nlohmann::json;

__constant__ AbsorberConfig d_absorber_config;

AbsorberConfig loadAbsorberConfigToGPU(const std::string& filepath){
    std::ifstream file(filepath);
    if(!file.is_open()) {
        std::cerr << "Failed to open config file: " << filepath << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    AbsorberConfig h_config;
    auto absorber_json = j["Absorber"];
    h_config.r = absorber_json["r"].get<float>();
    h_config.position.x = absorber_json["position"][0].get<float>();
    h_config.position.y = absorber_json["position"][1].get<float>();
    h_config.position.z = absorber_json["position"][2].get<float>();
    h_config.length = absorber_json["length"].get<float>();

    cudaError_t err = cudaMemcpyToSymbol(d_absorber_config, &h_config, sizeof(AbsorberConfig));
    if(err != cudaSuccess) {
        std::cerr << "Failed to copy absorber config to GPU symbol: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "Successfully loaded absorber config to GPU." << std::endl;
    return h_config;
}