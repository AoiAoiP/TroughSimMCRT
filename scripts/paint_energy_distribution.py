import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ==========================================
# 1. 配置参数 (请根据你的 config.json 确认)
# ==========================================
csv_file_path = "../build/flux_map.csv"  # 假设你在 scripts 目录下运行，csv 在 build 里
absorber_length = 1.9                  # 集热管总长度 (m)
absorber_radius = 0.04                 # 集热管半径 (m)

# ==========================================
# 2. 读取数据
# ==========================================
if not os.path.exists(csv_file_path):
    # 尝试在当前目录寻找
    csv_file_path = "flux_map.csv"
    if not os.path.exists(csv_file_path):
        print(f"❌ 错误: 找不到 {csv_file_path} 文件！请确认 CUDA 程序已成功运行并输出。")
        exit()

print(f"✅ 正在加载能流数据: {csv_file_path} ...")
# numpy 直接读取逗号分隔的 2D 数组
flux_map = np.loadtxt(csv_file_path, delimiter=',')

# 获取网格维度 (Z轴代表长度，X轴代表圆周角)
grid_res_z, grid_res_x = flux_map.shape
print(f"📊 网格解析完毕: Z轴(长度)={grid_res_z} 节点, X轴(圆周)={grid_res_x} 节点")

# ==========================================
# 3. 物理坐标系映射
# ==========================================
# 圆周总弧长
circumference = 2 * np.pi * absorber_radius

# 生成物理坐标网格 (用于精确绘制坐标轴)
x_coords = np.linspace(0, circumference, grid_res_x)
z_coords = np.linspace(-absorber_length/2, absorber_length/2, grid_res_z)
X, Z = np.meshgrid(x_coords, z_coords)

# ==========================================
# 4. 绘图: 2D 热力图 + 3D 表面图
# ==========================================
fig = plt.figure(figsize=(14, 8))

# --- 子图 1: 2D 展开热力图 ---
ax1 = fig.add_subplot(1, 2, 1)
# 使用 'hot' 或 'inferno' 颜色映射非常适合表示能流
im = ax1.imshow(flux_map, cmap='inferno', origin='lower', aspect='auto',
                extent=[0, circumference, -absorber_length/2, absorber_length/2])

# 添加等值线增强质感
ax1.contour(X, Z, flux_map, levels=8, colors='white', alpha=0.3, linewidths=0.8)

ax1.set_title('Absorber Flux Distribution (2D Unrolled)', fontsize=14)
ax1.set_xlabel('Circumferential Arc Length (m)', fontsize=12)
ax1.set_ylabel('Axial Length (m)', fontsize=12)
cbar = fig.colorbar(im, ax=ax1, pad=0.02)
cbar.set_label('Energy Flux Density (kW/m^2)', fontsize=12)

# --- 子图 2: 3D 能流峰值图 ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Z, flux_map, cmap='inferno', edgecolor='none', alpha=0.9)

ax2.set_title('3D Flux Peak Visualization', fontsize=14)
ax2.set_xlabel('Circumference (m)', fontsize=10)
ax2.set_ylabel('Axial Length (m)', fontsize=10)
ax2.set_zlabel('Flux Density', fontsize=10)

# 调整视角以便更好地观察峰值
ax2.view_init(elev=30, azim=45)

# ==========================================
# 5. 保存与展示
# ==========================================
plt.tight_layout()
save_path = "flux_distribution_result.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 绘图完成！图片已保存至: {save_path}")

# plt.show()