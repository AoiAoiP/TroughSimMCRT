import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CONFIG_PATH = '../resources/config.json'
EXEC_PATH = '../build/mcrt_sim'

# Focal length scan range: 1.2m to 2.7m, step ~0.05m
FOCAL_LENGTHS = np.linspace(1.2, 2.7, 31)

# Compare several typical slope error levels (unit: mrad)
SLOPE_ERRORS_MRAD = [0.0, 1.5, 2.0, 2.5, 3.0]

# Fixed absorber radius: DN80
FIXED_RADIUS = 0.04

def update_config(focal_length, slope_error):
    """Update focal length and absorber position in config.json"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 1. Set fixed radius and current slope error
    config['ParabolicTrough']['slope_error'] = slope_error

    # 2. Update parabolic trough focal length
    config['ParabolicTrough']['focal_length'] = focal_length

    # 3. Sync absorber position with focal length
    config['Absorber']['r'] = FIXED_RADIUS
    config['Absorber']['position'][2] = focal_length

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """Run CUDA program and parse intercept factor"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0

    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)',
                      result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val * 100.0 if val <= 1.0 else val
    return 0.0

def main():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(EXEC_PATH):
        print("Please check config.json or executable path.")
        return

    # Store curve data for different error levels
    results = {error: [] for error in SLOPE_ERRORS_MRAD}

    print("Starting focal length vs. slope error sensitivity scan...")

    for error_mrad in SLOPE_ERRORS_MRAD:
        print(f"\n--- Current slope error: {error_mrad} mrad ---")

        for fl in FOCAL_LENGTHS:
            update_config(fl, error_mrad)
            intercept = run_simulation()
            results[error_mrad].append(intercept)
            print(f"  Focal: {fl:.2f}m -> Intercept: {intercept:.2f}%")

    # --- Plot sensitivity curves ---
    plt.figure(figsize=(10, 6))

    colors = ['green', 'blue', 'yellow', 'red', 'orange']
    markers = ['o', 's', '^', 'D', 'v']

    for i, error_mrad in enumerate(SLOPE_ERRORS_MRAD):
        plt.plot(FOCAL_LENGTHS, results[error_mrad],
                 marker=markers[i], linestyle='-', color=colors[i],
                 label=f'Slope Error: {error_mrad} mrad')

    plt.title('Geometric Intercept Factor vs. Focal Length'
              ' at Different Errors', fontsize=14)
    plt.xlabel('Focal Length (m)', fontsize=12)
    plt.ylabel('Geometric Intercept Factor (%)', fontsize=12)

    # Tight Y range to highlight curvature variation
    plt.ylim(60, 105)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)

    plt.savefig('../out/focal_length_sensitivity.png', dpi=300)
    print("\nScan complete! Plot saved to '../out/focal_length_sensitivity.png'")
    # plt.show()

if __name__ == "__main__":
    main()