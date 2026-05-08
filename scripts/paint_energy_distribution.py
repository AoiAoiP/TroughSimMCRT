import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ==========================================
# 1. Configuration (confirm against your config.json)
# ==========================================
csv_file_path = "../out/flux_map.csv"   # assumes you run from scripts/ directory
absorber_length = 1.9                   # absorber tube total length (m)
absorber_radius = 0.04                  # absorber tube radius (m)

# ==========================================
# 2. Load data
# ==========================================
if not os.path.exists(csv_file_path):
    # try current directory as fallback
    csv_file_path = "flux_map.csv"
    if not os.path.exists(csv_file_path):
        print(f"Error: cannot find {csv_file_path}! "
              "Please ensure the CUDA program has run successfully.")
        exit()

print(f"Loading flux data: {csv_file_path} ...")
# numpy reads comma-separated 2D array
flux_map = np.loadtxt(csv_file_path, delimiter=',')

# extract grid dimensions (Z = axial, X = circumferential)
grid_res_z, grid_res_x = flux_map.shape
print(f"Grid parsed: axial(Z)={grid_res_z} nodes, circumferential(X)={grid_res_x} nodes")

# ==========================================
# 3. Physical coordinate mapping
# ==========================================
# total circumference of the absorber tube
circumference = 2 * np.pi * absorber_radius

# generate physical coordinate grid for accurate axis labeling
x_coords = np.linspace(0, circumference, grid_res_x)
z_coords = np.linspace(-absorber_length/2, absorber_length/2, grid_res_z)
X, Z = np.meshgrid(x_coords, z_coords)

# ==========================================
# 4. Plot: 2D heatmap + 3D surface
# ==========================================
fig = plt.figure(figsize=(14, 8))

# --- Subplot 1: 2D unrolled heatmap ---
ax1 = fig.add_subplot(1, 2, 1)
# 'inferno' colormap works well for flux
im = ax1.imshow(flux_map, cmap='inferno', origin='lower', aspect='auto',
                extent=[0, circumference, -absorber_length/2, absorber_length/2])

# overlay contour lines for visual enhancement
ax1.contour(X, Z, flux_map, levels=8, colors='white', alpha=0.3, linewidths=0.8)

ax1.set_title('Absorber Flux Distribution (2D Unrolled)', fontsize=14)
ax1.set_xlabel('Circumferential Arc Length (m)', fontsize=12)
ax1.set_ylabel('Axial Length (m)', fontsize=12)
cbar = fig.colorbar(im, ax=ax1, pad=0.02)
cbar.set_label('Energy Flux Density (kW/m^2)', fontsize=12)

# --- Subplot 2: 3D flux surface ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Z, flux_map, cmap='inferno', edgecolor='none', alpha=0.9)

ax2.set_title('3D Flux Peak Visualization', fontsize=14)
ax2.set_xlabel('Circumference (m)', fontsize=10)
ax2.set_ylabel('Axial Length (m)', fontsize=10)
ax2.set_zlabel('Flux Density', fontsize=10)

# adjust view angle to better show peaks
ax2.view_init(elev=30, azim=45)

# ==========================================
# 5. Save and display
# ==========================================
plt.tight_layout()
save_path = "../out/flux_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot complete! Saved to: {save_path}")

# plt.show()