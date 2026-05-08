#!/bin/bash

set -e

echo "========================================"
echo " 1. Building CUDA MCRT engine..."
echo "========================================"
cd build
make -j$(nproc)

echo ""
echo "========================================"
echo " 2. Running flux simulation..."
echo "========================================"
# Output: flux_map.csv saved under out/ directory
./mcrt_sim

echo ""
echo "========================================"
echo " 3. Generating 2D/3D flux density plots..."
echo "========================================"
python ../scripts/paint_energy_distribution.py

echo ""
echo "========================================"
echo " 4. Running analysis scripts..."
echo "========================================"
python ../scripts/scan_intercept.py
python ../scripts/scan_focal_length.py
python ../scripts/find_optimal_focal.py

cd ..

echo ""
echo "========================================"
echo " Pipeline complete! Check the out/ directory for results."
echo "========================================"
