#include "config/sun_config.cuh"

__constant__ SunConfig d_sun_config;

using json = nlohmann::json;

SunShapeType parseSunShapeType(const std::string& type_str) {
    if (type_str == "UNIFORM") return SunShapeType::UNIFORM;
    if (type_str == "GAUSSIAN") return SunShapeType::GAUSSIAN;
    if (type_str == "BUIE") return SunShapeType::BUIE;
    if (type_str == "DEFINED") return SunShapeType::DEFINED;
    throw std::invalid_argument("Unknown Sun Shape Type: " + type_str);
}

SunConfig loadSunConfigToGPU(const std::string& filepath) {
    std::ifstream file(filepath);
    if(!file.is_open()) {
        std::cerr << "Failed to open config file: " << filepath << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    SunConfig h_config;
    auto sun_json = j["Sun"];
    h_config.azimuth = sun_json["azimuth"].get<float>();
    h_config.zenith = sun_json["zenith"].get<float>();
    h_config.DNI = sun_json["DNI"].get<float>();
    h_config.dir.x = sun_json["dir"][0].get<float>();
    h_config.dir.y = sun_json["dir"][1].get<float>();
    h_config.dir.z = sun_json["dir"][2].get<float>();

    std::string shape_str = sun_json["shape"].get<std::string>();
    h_config.sunshape = parseSunShapeType(shape_str);

    h_config.d_cdf_angles = nullptr;
    h_config.d_cdf_values = nullptr;
    h_config.table_size = 0;

    if(h_config.sunshape == SunShapeType::UNIFORM){
        h_config.params.theta_max = sun_json["theta_max"].get<float>();
    }else if(h_config.sunshape == SunShapeType::GAUSSIAN){
        h_config.params.sigma = sun_json["sigma"].get<float>();
    }else if(h_config.sunshape == SunShapeType::DEFINED || h_config.sunshape == SunShapeType::BUIE) {
        // If BUIE/DEFINED shape, load CDF table
        std::string csv_path = sun_json["csv_path"].get<std::string>();
        std::ifstream csv_file(csv_path);

        std::vector<float> angles;
        std::vector<float> intensities;
        std::string line;

        while(std::getline(csv_file,line)){
            std::stringstream ss(line);
            std::string cell;

            std::getline(ss,cell,',');
            angles.push_back(std::stof(cell));

            std::getline(ss,cell,',');
            intensities.push_back(std::stof(cell));
        }

        int N = angles.size();
        h_config.table_size = N;

        std::vector<float> cdf(N,0.0f);
        float sum = 0.0f;
        for(int i = 1; i < N; ++i) {
            float theta_prev = angles[i -1] * 0.001f;
            float theta_curr = angles[i] * 0.001f;
            float d_theta = theta_curr - theta_prev;
            float avg_intensity = (intensities[i-1] + intensities[i]) / 2.0f;
            float avg_sin_theta = (sinf(theta_prev) + sinf(theta_curr)) / 2.0f;
            float step_val = avg_intensity * 2.0f * PI * avg_sin_theta * d_theta;
            sum += step_val;
            cdf[i] = sum;
        }
        cdf[0]=0.0f;

        // 归一化
        for(int i = 0; i < N; ++i) {
            cdf[i] /= sum;
            // std::cout<<"Angle: "<<angles[i]<<" mrad, Intensity: "<<intensities[i]<<", CDF: "<<cdf[i]<<std::endl;
        }

        cudaMalloc(&h_config.d_cdf_angles, N * sizeof(float));
        cudaMalloc(&h_config.d_cdf_values, N * sizeof(float));
        cudaMemcpy(h_config.d_cdf_angles, angles.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(h_config.d_cdf_values, cdf.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "Loaded CSV sun shape from " << csv_path << " with " << N << " data points." << std::endl;
    }

    cudaError_t err = cudaMemcpyToSymbol(d_sun_config, &h_config, sizeof(SunConfig));
    if(err != cudaSuccess) {
        std::cerr << "Failed to copy config to GPU symbol: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "Successfully loaded sun config to GPU.\n";
    return h_config;
}