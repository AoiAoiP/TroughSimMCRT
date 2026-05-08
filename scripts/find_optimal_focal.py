import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import copy

# --- Core engineering parameter configuration ---
CONFIG_PATH = '../resources/config.json'
EXEC_PATH = '../build/mcrt_sim'

# Trough aperture width
DEFALUT_WIDTH = 8.61502
# APERTURE_WIDTH = np.linspace(5.6, 9.1, 8)
APERTURE_WIDTH = 8.6
SLOPE_ERRORS_RAD = np.linspace(0.5, 3, 6)
# SLOPE_ERROR_RAD = 2
SPEC_ERROR_RAD = 2.5

# Fine focal length scan: 1.0m to 3.0m, step 0.05m (41 points)
FOCAL_LENGTHS = np.linspace(1.0, 3.0, 41)

def update_config(focal_length, slope_error, original_config):
    """Update config.json with fixed boundary conditions,
       sync focal length with absorber position"""
    config = copy.deepcopy(original_config)
    # 1. Set fixed boundary condition
    config['ParabolicTrough']['slope_error'] = slope_error

    # 2. Sync focal length and absorber centre height
    config['ParabolicTrough']['focal_length'] = focal_length
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

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    try:
        results = {}  # store results per slope error level

        print(f"Starting multi-parameter optimisation..."
              f" (slope error range: {SLOPE_ERRORS_RAD[0]:.1f} - {SLOPE_ERRORS_RAD[-1]:.1f} mrad)")

        # --- Nested loop scan ---
        for s in SLOPE_ERRORS_RAD:
            print(f"\n>> Testing slope error: {s:.2f} mrad")
            results[s] = {'intercepts': [], 'rim_angles': []}

            for fl in FOCAL_LENGTHS:
                update_config(fl, s, original_config)
                intercept = run_simulation()
                rim_angle_deg = np.degrees(
                    2 * np.arctan(APERTURE_WIDTH / (4 * fl)))

                results[s]['intercepts'].append(intercept)
                results[s]['rim_angles'].append(rim_angle_deg)
                print(f"  F={fl:.2f}m -> IF={intercept:.2f}%, "
                      f"Rim={rim_angle_deg:.1f} deg")

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()

        # Viridis colour map
        colors = cm.viridis(np.linspace(0, 0.9, len(SLOPE_ERRORS_RAD)))

        optimal_stats = []

        for i, s in enumerate(SLOPE_ERRORS_RAD):
            c = colors[i]
            intercepts = results[s]['intercepts']
            rim_angles = results[s]['rim_angles']

            # Record optimum for each slope error level
            max_if = max(intercepts)
            opt_idx = intercepts.index(max_if)
            opt_fl = FOCAL_LENGTHS[opt_idx]
            opt_rim = rim_angles[opt_idx]
            optimal_stats.append(
                f"Slope={s:.1f}mrad: IF={max_if:.1f}%, "
                f"F={opt_fl:.2f}m (Rim: {opt_rim:.1f} deg)")

            # Left axis (intercept factor): solid line
            ax1.plot(FOCAL_LENGTHS, intercepts, marker='.', linestyle='-',
                     color=c, linewidth=2,
                     label=f'IF (Slope={s:.1f} mrad)')
            # Mark optimum
            ax1.plot(opt_fl, max_if, marker='*', color=c, markersize=12)

            # Right axis (rim angle): dashed line
            ax2.plot(FOCAL_LENGTHS, rim_angles, linestyle='--',
                     color=c, alpha=0.6)

        # Axis labels
        ax1.set_xlabel('Focal Length (m)', fontsize=12)
        ax1.set_ylabel('Geometric Intercept Factor (%)', fontsize=12)
        ax2.set_ylabel('Rim Angle (degrees)', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # Recommended rim angle range on right axis
        ax2.axhspan(80, 90, color='orange', alpha=0.15,
                    label='Recommended Rim Angle (80-90 deg)')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='lower center', bbox_to_anchor=(0.5, -0.25),
                   ncol=3, fontsize=10)

        plt.title('Intercept Factor & Rim Angle vs Focal Length'
                  ' (Varying Slope Error)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # leave room for bottom legend

        plt.savefig('../out/multi_width_focal_length.png', dpi=300)
        print("\nOptimisation complete! "
              "Plot saved to '../out/multi_width_focal_length.png'")
        print("\nFocal length vs. intercept factor summary:")
        for stat in optimal_stats:
            print("  " + stat)

    finally:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(original_config, f, indent=4)
        print("\nConfig restored to original state.")

if __name__ == "__main__":
    main()