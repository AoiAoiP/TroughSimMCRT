#include "config/trough_config.cuh"
#include <vector>
#include <string>
#include <cstring>

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

    // --- torsion error parsing ---
    h_config.torsion_error.type = TorsionErrorConfig::NONE;
    h_config.torsion_error.y_pos = nullptr;
    h_config.torsion_error.torsion_values = nullptr;
    h_config.torsion_error.table_size = 0;
    std::memset(h_config.torsion_error.coefficients, 0, sizeof(h_config.torsion_error.coefficients));

    if (pt_json.contains("torsion_error")) {
        auto te_json = pt_json["torsion_error"];
        std::string te_type_str = te_json.value("type", "none");

        if (te_type_str == "polynomial") {
            h_config.torsion_error.type = TorsionErrorConfig::POLYNOMIAL;
            if (te_json.contains("coefficients")) {
                auto coeffs = te_json["coefficients"];
                int n = std::min((int)coeffs.size(), 4);
                for (int i = 0; i < n; ++i) {
                    h_config.torsion_error.coefficients[i] = coeffs[i].get<float>();
                }
            }
        } else if (te_type_str == "lookup") {
            h_config.torsion_error.type = TorsionErrorConfig::LOOKUP;
            auto y_positions = te_json["y_positions"];
            auto torsion_values = te_json["torsion_values"];
            int N = y_positions.size();

            std::vector<float> h_y_pos(N);
            std::vector<float> h_torsion(N);
            for (int i = 0; i < N; ++i) {
                h_y_pos[i] = y_positions[i].get<float>();
                h_torsion[i] = torsion_values[i].get<float>();
            }

            h_config.torsion_error.table_size = N;
            cudaMalloc(&h_config.torsion_error.y_pos, N * sizeof(float));
            cudaMalloc(&h_config.torsion_error.torsion_values, N * sizeof(float));
            cudaMemcpy(h_config.torsion_error.y_pos, h_y_pos.data(), N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(h_config.torsion_error.torsion_values, h_torsion.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        }
        // "none" or unknown: keep defaults (NONE, nullptrs, zeros)
    }

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