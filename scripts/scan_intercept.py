import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
CONFIG_PATH = '../resources/config.json'   # adjust to your actual path
EXEC_PATH = '../build/mcrt_sim'            # path to the CUDA executable
SLOPE_ERRORS_MRAD = np.linspace(0, 5, 11)  # scan range: 0 to 5 mrad, 11 points
TUBES = {
    'DN80': 0.04,   # radius 40 mm
    'DN90': 0.045   # radius 45 mm
}

def update_config(radius, slope_error):
    """Read and update config.json file"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    config['Absorber']['r'] = radius
    config['Absorber']['position'][2] = 1.7
    config['ParabolicTrough']['slope_error'] = slope_error
    config['ParabolicTrough']['focal_length'] = 1.7

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """Run CUDA program and parse intercept factor from stdout"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"CUDA program failed:\n{result.stderr}")
        return 0.0

    output = result.stdout

    # --- Extract intercept factor from CUDA log ---
    # Example: "Intercept Factor: 0.985"
    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)', output, re.IGNORECASE)

    if match:
        intercept_factor = float(match.group(1))
        if intercept_factor <= 1.0:
            intercept_factor *= 100.0
        return intercept_factor
    else:
        print("Could not find intercept factor in output."
              "Check the regex or program output.")
        print("Last 200 characters of output:", output[-200:])
        return 0.0

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found: {CONFIG_PATH}")
        return
    if not os.path.exists(EXEC_PATH):
        print(f"Executable not found: {EXEC_PATH}. Please compile the C++ project first.")
        return

    results = {name: [] for name in TUBES.keys()}

    print("Starting automated MCRT simulation scan...")

    for name, radius in TUBES.items():
        print(f"\n--- Testing absorber: {name} (r={radius}m) ---")
        for error in SLOPE_ERRORS_MRAD:
            # 1. Update config
            update_config(radius, error)

            # 2. Run and capture result
            intercept = run_simulation()
            results[name].append(intercept)

            print(f"Slope error: {error:.1f} mrad | Intercept factor: {intercept:.2f}%")

    # --- Plot comparison ---
    plt.figure(figsize=(10, 6))

    # Plot DN80 and DN90 curves
    plt.plot(SLOPE_ERRORS_MRAD, results['DN80'], marker='o', linestyle='-',
             color='b', label='DN80 (r=40mm)')
    plt.plot(SLOPE_ERRORS_MRAD, results['DN90'], marker='s', linestyle='--',
             color='r', label='DN90 (r=45mm)')

    plt.title('Geometric Intercept Factor vs. Slope Error', fontsize=14)
    plt.xlabel('Slope Error (mrad)', fontsize=12)
    plt.ylabel('Geometric Intercept Factor (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)

    # Save figure
    plt.savefig('../out/intercept_factor_comparison.png', dpi=300)
    print("\nScan complete! Plot saved to '../out/intercept_factor_comparison.png'")
    # plt.show()

if __name__ == "__main__":
    main()