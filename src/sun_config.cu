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

    std::string shape_str = sun_json["shape"].get<std::string>();
    h_config.sunshape = parseSunShapeType(shape_str);

    h_config.d_cdf_angles = nullptr;
    h_config.d_cdf_values = nullptr;
    h_config.table_size = 0;

    switch(h_config.sunshape) {
        case SunShapeType::UNIFORM:
            h_config.params.theta_max = sun_json["theta_max"].get<float>();
            break;
        case SunShapeType::GAUSSIAN:
            h_config.params.sigma = sun_json["sigma"].get<float>();
            break;
        case SunShapeType::BUIE:
            h_config.params.csr = sun_json["csr"].get<float>();
            break;
        case SunShapeType::DEFINED:
            // For DEFINED shape, we will load the CDF table later
            break;
    }

    // If DEFINED shape, load CDF table
    if(h_config.sunshape == SunShapeType::DEFINED) {
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
        for(int i = 0; i < N; ++i) {
            float d_theta_rad = (i == 0) ? angles[i] * 0.001f : (angles[i] - angles[i-1]) * 0.001f; // Convert mrad to rad
            float theta_rad = angles[i] * 0.001f; // Convert mrad to rad
            float step_val = intensities[i] * sinf(theta_rad) * d_theta_rad; // I(θ) * sin(θ) * dθ
            sum += step_val;
            cdf[i] = sum;
        }

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

__host__ void printSunInfo(const SunConfig& config){
        // printf("Sun Shape: %d\n", config.sunshape);
        printf("Zenith: %.2f degrees\n", config.zenith);
        printf("Azimuth: %.2f degrees\n", config.azimuth);
        switch(config.sunshape){
            case SunShapeType::UNIFORM:
                printf("Uniform Sunshape with theta_max = %.4f radians\n", config.params.theta_max);
                break;
            case SunShapeType::GAUSSIAN:
                printf("Gaussian Sunshape with sigma = %.4f radians\n", config.params.sigma);
                break;
            case SunShapeType::BUIE:
                printf("Buie Sunshape with CSR = %.4f\n", config.params.csr);
                break;
            case SunShapeType::DEFINED:
                printf("Defined Sunshape with CDF table size = %d\n", config.table_size);
                break;
    }
}