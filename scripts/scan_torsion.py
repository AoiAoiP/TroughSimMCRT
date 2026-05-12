import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

CONFIG_PATH = '../resources/Nevada_Solar_One/config.json'
EXEC_PATH = '../build/mcrt_sim'
TORSION_VALS_MRAD = np.linspace(0, 30, 31)  # 0 to 30 mrad, 31 points

def update_config(torsion_mrad):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    te = config['ParabolicTrough']['torsion_error']
    n = len(te['y_positions'])
    te['type'] = 'lookup'
    te['torsion_values'] = [torsion_mrad] * n

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Simulation failed:\n{result.stderr}")
        return 0.0

    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)',
                      result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val * 100.0 if val <= 1.0 else val
    return 0.0

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}")
        return
    if not os.path.exists(EXEC_PATH):
        print(f"Executable not found: {EXEC_PATH}")
        return

    intercepts = []

    print("Scanning torsion error 0 – 30 mrad...")
    for t in TORSION_VALS_MRAD:
        update_config(t)
        intercept = run_simulation()
        intercepts.append(intercept)
        print(f"  Torsion: {t:.1f} mrad -> Intercept: {intercept:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(TORSION_VALS_MRAD, intercepts, marker='o', linestyle='-', color='teal')
    plt.title('Intercept Factor vs. Torsion Error', fontsize=14)
    plt.xlabel('Torsion Error (mrad)', fontsize=12)
    plt.ylabel('Intercept Factor (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig('../out/torsion_sensitivity.png', dpi=300)
    print("\nDone. Plot saved to '../out/torsion_sensitivity.png'")

if __name__ == "__main__":
    main()
