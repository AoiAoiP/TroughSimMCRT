import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline

# 网格参数
# num_theta = 52  # 周向分段数
# num_y = 23*42      # 轴向分段数

def analyze_energy_distribution(csv_file,num_theta = 52,num_y = 23*42):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"文件: {csv_file}")
    print(f"数据范围: seg_th从{df['seg_th'].min()}到{df['seg_th'].max()}")
    print(f"总数据点数: {len(df)}")
    print(f"期望数据点数: {num_theta * num_y}")

    # 创建能量矩阵
    energy = np.zeros((num_y, num_theta))

    # 填充矩阵
    for index, row in df.iterrows():
        seg_idx = int(row['seg_th'])
        if seg_idx < num_y * num_theta:
            y_idx = seg_idx // num_theta  # 轴向索引
            theta_idx = seg_idx % num_theta   # 周向索引
            energy[y_idx, theta_idx] = row['part_energy(W/m2)']

    print("矩阵填充完成")
    # print(f"矩阵形状: {energy.shape}")
    print(f"最小值: {energy.min():.6f}")
    print(f"最大值: {energy.max():.6f}")
    print(f"平均值: {energy.mean():.6f}")

    mid_point = num_theta // 2

    reordered_matrix = np.zeros_like(energy)
    for i in range(num_y):
        # 重新排列周向顺序：将整个圆周完整显示
        for j in range(num_theta):
            new_theta_idx = (j) % num_theta
            reordered_matrix[i, new_theta_idx] = energy[i, j]

    # 创建自定义颜色映射
    colors = ['navy', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # 设置绘图风格
    plt.style.use('default')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # 绘制热力图 - 显示整个圆周
    im = ax.imshow(reordered_matrix, cmap=cmap, aspect='auto', origin='lower',
                extent=[-mid_point, mid_point, 0, num_y-1])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Energy(W)', fontsize=14, fontweight='bold')

    # 设置坐标轴标签
    ax.set_xlabel('Circumferential Position (θ/°)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Axial Position (y/m)', fontsize=14, fontweight='bold')
    ax.set_title(f'MCRT Simulation by CUDA\nSlope Error=2.5mrad', 
                fontsize=16, fontweight='bold', pad=20)

    # 设置实际角度刻度：-180° ~ 180°，步长 30°
    theta_angles = np.arange(-180, 181, 30)

    # 将真实角度转换为矩阵的 x 坐标
    theta_ticks = theta_angles * (num_theta / 360.0)

    # 设置到坐标轴
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([f"{ang}°" for ang in theta_angles])

    # 设置轴向刻度
    tube_length = 23.96
    y_step = tube_length / num_y
    y_ticks = np.arange(0, num_y, 50) 
    y_positions = y_ticks * y_step - tube_length / 2
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.2f}' for y in y_positions])

    # 添加重要角度参考线
    important_angles = [-180,-120,-60,0,60,120,180]
    for angle in important_angles:
        x_pos = angle * (num_theta / 360)
        if -mid_point <= x_pos <= mid_point:
            ax.axvline(x=x_pos, color='white', linestyle='--', linewidth=1, alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'Distribution_from_{csv_file.split(".")[0]}.png', dpi=300, bbox_inches='tight', facecolor='white')
    # plt.show()

analyze_energy_distribution('CUDA_MCRT_energy_absorbed_origin.csv')
analyze_energy_distribution('CUDA_MCRT_energy_absorbed_filtered.csv')

# 创建归一化对象
# norm = plt.Normalize(vmin=reordered_matrix.min(), vmax=reordered_matrix.max())

# 绘制3D图
# drawflux3dnew(energy, norm)