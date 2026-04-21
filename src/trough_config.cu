#include "config/trough_config.cuh"

using json = nlohmann::json;

__constant__ ParabolicTroughConfig d_trough_config;

ParabolicTroughConfig loadTroughConfigToGPU(const std::string& filepath) {
    std::ifstream file(filepath);
    if(!file.is_open()) {
        std::cerr << "Failed to open config file: " << filepath << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    ParabolicTroughConfig h_config;
    auto pt_json = j["ParabolicTrough"];
    h_config.focal_length = pt_json["focal_length"].get<float>();
    h_config.length = pt_json["length"].get<float>();
    h_config.position.x = pt_json["position"][0].get<float>();
    h_config.position.y = pt_json["position"][1].get<float>();
    h_config.position.z = pt_json["position"][2].get<float>();
    h_config.width = pt_json["width"].get<float>();
    h_config.reflectivity = pt_json["reflectivity"].get<float>();
    h_config.slope_error = pt_json["slope_error"].get<float>();
    h_config.specularity_error = pt_json["specularity_error"].get<float>();
    h_config.valid_width = 0;
    auto bounds = pt_json["bounds"];
    for(int i = 0; i < NUM_SUB_MIRRORS; ++i) {
        h_config.bounds[i].x = bounds[i][0].get<float>();
        h_config.bounds[i].y = bounds[i][1].get<float>();
        h_config.valid_width += (h_config.bounds[i].y - h_config.bounds[i].x);
    }

    cudaError_t err = cudaMemcpyToSymbol(d_trough_config, &h_config, sizeof(ParabolicTroughConfig));
    if(err != cudaSuccess) {
        std::cerr << "Failed to copy config to GPU symbol: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "Successfully loaded trough config to GPU." << std::endl;
    return h_config;
}