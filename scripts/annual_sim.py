import json
import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- Configuration ---
CONFIG_PATH = '../resources/config.json'
EXEC_PATH = '../build/mcrt_sim'
SOLAR_DATA_TXT = '../resources/738_sundir_year.txt'  # sun direction data file

# --- Absorber tube diameters under test ---
TUBES = {
    'DN80': 0.040,   # radius 40 mm
    'DN90': 0.045    # radius 45 mm
}

def update_config_sun_dir(sun_vector, radius):
    """Update sun direction and absorber radius in config.json"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 1. Update absorber radius
    config['Absorber']['r'] = radius

    # 2. Update sun direction vector
    config['Sun']['dir'] = [sun_vector[0], sun_vector[1], sun_vector[2]]

    # 3. Extend trough/absorber length to suppress end effects
    config['ParabolicTrough']['length'] = 1000
    config['ParabolicTrough']['position'][2] = 500
    config['Absorber']['position'][2] = 500
    config['Absorber']['length'] = 1001

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """Run CUDA MCRT program and capture geometric intercept factor"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0

    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)',
                      result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val if val > 1.0 else val * 100.0
    return 0.0

def main():
    if not os.path.exists(SOLAR_DATA_TXT):
        print(f"Error: annual solar data file not found: {SOLAR_DATA_TXT}")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    try:
        # Read sun direction vectors; columns: x, y, z, unused
        df = pd.read_csv(SOLAR_DATA_TXT, sep=r'\s+', header=None,
                         names=['x', 'y', 'z', 'unused'])

        # Default DNI to constant 1000 W/m2 if not present
        if 'dni' not in df.columns:
            df['dni'] = 1000.0

        print(f"Loaded annual solar data, {len(df)} time steps.")

        annual_results = {}

        for tube_name, radius in TUBES.items():
            print(f"\nStarting {tube_name} (r={radius}m) "
                  f"annual energy-weighted intercept calculation...")

            total_energy_hit = 0.0
            total_energy_available = 0.0
            valid_hours = 0

            start_time = time.time()

            for index, row in df.iterrows():
                x, z, y = row['x'], row['y'], row['z']
                dni = 1000

                # Normalize sun direction vector
                norm = np.sqrt(x**2 + y**2 + z**2)
                gx, gy, gz = x/norm, y/norm, z/norm

                # Trough length along Y axis. Tracking system rotates
                # around Y to zero out the X component.
                local_x = 0.0
                local_y = gy
                local_z = np.sqrt(gx**2 + gz**2)
                sun_vec = [local_x, local_y, local_z]
                cos_theta = local_z

                # Update JSON and run simulation
                update_config_sun_dir(sun_vec, radius)
                intercept_factor_pct = run_simulation()

                # Energy-weighted accumulation
                available_energy = dni * cos_theta
                hit_energy = available_energy * (intercept_factor_pct / 100.0)
                print(f"row:{valid_hours}, "
                      f"intercept_factor_pct:{intercept_factor_pct:.2f}")

                total_energy_available += available_energy
                total_energy_hit += hit_energy
                valid_hours += 1

                # Progress indicator
                if valid_hours % 500 == 0:
                    print(f"  Completed {valid_hours} valid hours...")

            # Final annual weighted intercept efficiency
            if total_energy_available > 0:
                annual_efficiency = (
                    total_energy_hit / total_energy_available) * 100.0
            else:
                annual_efficiency = 0.0

            annual_results[tube_name] = annual_efficiency

            cost_time = time.time() - start_time
            print(f"{tube_name} complete! Time: {cost_time:.1f} s")
            print(f"{tube_name} annual energy-weighted intercept "
                  f"efficiency: {annual_efficiency:.2f}%")

        # --- Comparison bar chart ---
        plt.figure(figsize=(8, 6))
        bars = plt.bar(annual_results.keys(), annual_results.values(),
                       color=['#1f77b4', '#ff7f0e'], width=0.5)

        # Add data labels
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
                     f'{yval:.2f}%', ha='center', va='bottom',
                     fontsize=12, fontweight='bold')

        plt.title('Annual Energy-Weighted Intercept Factor\n'
                  '(DN80 vs DN90)', fontsize=14)
        plt.ylabel('Annual Intercept Efficiency (%)', fontsize=12)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        plt.savefig('annual_efficiency_comparison.png', dpi=300)
        print("\nPlot saved to 'annual_efficiency_comparison.png'")
        plt.show()
    finally:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(original_config, f, indent=4)
        print("\nConfig restored to original state.")

if __name__ == "__main__":
    main()