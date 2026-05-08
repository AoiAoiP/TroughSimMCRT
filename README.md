# Parabolic Trough Solar Collector — CUDA-Accelerated MCRT Thermal Simulation

A high-performance GPU-accelerated Monte Carlo Ray Tracing (MCRT) solver for computing the flux density distribution on the outer surface of absorber tubes in Parabolic Trough Solar Collectors (PTSC).

## Overview

This solver uses **backward ray tracing**: rays originate from the absorber tube, reflect off the parabolic mirror segments toward the sun direction. Only rays that actually reach the absorber are counted, making the computation efficient and physically accurate.

## Key Features

- **GPU-accelerated**: Pure CUDA C++ implementation supporting tens of millions of rays with millisecond-level simulation times.
- **Pre-generated random pools**: MT19937-based random float2 pairs and per-ray random start indices (RSIA) are precomputed on the CPU and stored in GPU global memory, avoiding expensive `curand` calls inside the kernel.
- **Accurate geometry modeling**:
  - Robust quadratic root solver using `-0.5·(b ± √D)` formulation to avoid floating-point catastrophic cancellation.
  - Efficient inline ray-paraboloid and ray-cylinder intersection routines.
  - 6-segment sub-mirror support with boundary clipping.
- **Multiple sun shape models**: Uniform, Gaussian, Buie (CSR), and custom-defined sun shape via CSV lookup tables.
- **Gaussian post-processing**: A 5×5 separable Gaussian filter kernel smooths Monte Carlo noise. The circumferential direction uses periodic boundary conditions (cylinder wrap-around); the axial direction uses clamped boundaries.
- **Physical output**: Flux density calibrated to kW/m², intercept factor, and optical efficiency are computed and reported.

## Requirements

- **Host compiler**: GCC 9.0+ with C++17 support (or MSVC)
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.18 or later
- **Python**: 3.8+ (for visualization and parameter-scan scripts)
- **Python packages**: `numpy`, `matplotlib` (and optionally `pandas` for annual simulation)

## Quick start

```bash
# Clone the repository
git clone https://github.com/AoiAoiP/TroughSimMCRT.git
cd TroughSimMCRT

# Build
mkdir build && cd build
cmake .. && make -j$(nproc)

# Run simulation (single shot)
./mcrt_sim

# Or run the full pipeline (build + sim + plots + analysis)
cd .. && ./run.sh
```

The simulation reads `resources/config.json` and writes the flux map CSV to the `out/` directory.

## Repository structure

```
.
├── CMakeLists.txt                  # CMake build configuration (CUDA SM 86)
├── run.sh                          # One-shot pipeline script
├── README.md
├── CLAUDE.md                       # Project development guide (Chinese)
├── resources/
│   ├── config.json                 # Main simulation configuration
│   ├── CSR-5.csv                   # Buie sun shape data (if applicable)
│   └── 738_sundir_year.txt         # Annual sun direction data
├── include/
│   ├── app.cuh                     # Master header; SamplingDimension enum
│   ├── cu_math.cuh                 # CUDA vector math library (float2/3/4 operators)
│   ├── config/
│   │   ├── json.hpp                # nlohmann JSON library (single header)
│   │   ├── trough_config.cuh       # ParabolicTroughConfig struct + loader
│   │   ├── absorber_config.cuh     # AbsorberConfig struct + loader
│   │   ├── sun_config.cuh          # SunConfig struct + loader + SunShapeType enum
│   │   ├── sim_config.cuh          # SimConfig struct + loader
│   │   └── sample_vMF.cuh          # von Mises-Fisher sampling (unused by kernel)
│   ├── geometry/
│   │   ├── geometry_func.cuh       # HitInfo struct + robust solve_quadratic()
│   │   ├── trough_intersect.cuh    # sampleTrough() + intersectTrough()
│   │   └── absorber_intersect.cuh  # intersectAbsorber()
│   ├── optics/
│   │   ├── random_pools.cuh        # Random pool declarations + hash_index() + get_random_pair()
│   │   ├── sun_shape.cuh           # SampleSunshape() + SampleThetaFromcdf()
│   │   └── surface_error.cuh       # GaussianPerturb() (Box-Muller)
│   └── postprocess/
│       └── gaussian_filter.cuh     # 5×5 Gaussian filter kernel + host wrapper
├── src/
│   ├── main.cu                     # render() kernel + main() pipeline
│   ├── trough_config.cu            # loadTroughConfigToGPU()
│   ├── sun_config.cu               # loadSunConfigToGPU()
│   ├── absorber_config.cu          # loadAbsorberConfigToGPU()
│   ├── sim_config.cu               # loadSimConfigToGPU()
│   └── random_pools.cu             # initRandomPools() + freeRandomPools()
└── scripts/
    ├── paint_energy_distribution.py # 2D heatmap + 3D surface plot
    ├── scan_intercept.py            # Slope error × tube diameter parameter scan
    ├── scan_focal_length.py         # Focal length sensitivity scan
    ├── find_optimal_focal.py        # Multi-aperture × focal length joint optimisation
    └── annual_sim.py                # Annual energy-weighted intercept factor
```

## Configuration reference (`resources/config.json`)

### Simulation
| Key | Description | Typical value |
|-----|-------------|---------------|
| `total_rays` | Number of Monte Carlo rays | `10000000` |
| `block_size` | CUDA thread block size | `256` |
| `grid_res_z` | Axial grid bins | `100` |
| `grid_res_x` | Circumferential grid bins | `60` |

### ParabolicTrough
| Key | Description | Typical value |
|-----|-------------|---------------|
| `focal_length` | Focal length (m) | `1.7` |
| `length` | Trough length (m) | `1.8` |
| `width` | Gross aperture width (m) | `8.615` |
| `reflectivity` | Mirror reflectivity | `0.92` |
| `slope_error` | RMS slope error (mrad) | `2.0` |
| `specularity_error` | RMS specularity error (mrad) | `2.5` |
| `bounds` | 6 sub-mirror X intervals | See config |

### Absorber
| Key | Description | Typical value |
|-----|-------------|---------------|
| `r` | Tube radius (m) | `0.045` |
| `position` | Tube centre [X, Y, Z] (m) | `[0.019, 0.9, 1.7]` |
| `length` | Tube length (m) | `1.9` |

### Sun
| Key | Description | Typical value |
|-----|-------------|---------------|
| `azimuth` / `zenith` | Sun position (deg) | `180.0` / `0.0` |
| `dir` | Sun direction vector | `[0, 0, 1]` |
| `DNI` | Direct normal irradiance (W/m²) | `1000.0` |
| `shape` | Sun shape model | `UNIFORM`, `GAUSSIAN`, `BUIE`, `DEFINED` |
| `csv_path` | Path to CSV sun shape data (Buie/Defined) | `../resources/CSR-5.csv` |

## Coordinate convention

- **X**: Parabolic trough width (horizontal), following `z = x² / (4f)`
- **Y**: Trough/tube length direction (axial)
- **Z**: Vertical / parabolic depth direction

The trough mirror centre is at Y = 0.9 m; the absorber tube centre is at (X = 0.019, Y = 0.9, Z = 1.7) m.

## Physical model

The total equivalent surface error is computed as:

```
σ_total = sqrt(4 × σ_slope² + σ_specularity²)  [mrad → rad: × 0.001]
```

Flux calibration:

```
flux [kW/m²] = hits × ρ_reflectivity × DNI × A_aperture / (N_rays × A_bin)
```

where `A_bin = 2πr × L_tube / (grid_res_x × grid_res_z)`.

Circumferential binning uses `atan2(P.x - centre.x, P.z - centre.z) / 2π`, wrapping around [0, 2π).

## Known issues

- The flux summation energy computation in `main.cu` (lines 117–125) has a dimensional inconsistency: it sums flux (kW/m²) directly as if it were energy.
- `SampleThetaFromcdf` uses binary search over a CDF table, which causes warp-divergent branching. A uniform-resampling lookup table would be faster.
- The statistical quality of `hash_index` in `random_pools.cuh` has not been rigorously verified.
- The kernel lacks `cudaDeviceSynchronize()` for catching asynchronous execution errors.

## Building notes

- CUDA target architecture: SM 86 (RTX 30/40 series). Adjust `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU.
- `-use_fast_math` is enabled for the CUDA compiler — this trades a tiny amount of floating-point accuracy for significant speedups in transcendental functions (sin, cos, sqrt, etc.).

## License

This project is for research purposes. Contact the author for licensing details.
