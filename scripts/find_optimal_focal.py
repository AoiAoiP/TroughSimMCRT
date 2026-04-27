import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import copy

# --- 核心工程物理参数配置 ---
CONFIG_PATH = '../resources/config.json'   
EXEC_PATH = '../build/mcrt_sim'          

# 槽式开口宽度
DEFALUT_WIDTH = 8.61502
APERTURE_WIDTH = np.linspace(5.6, 9.1, 8)
ABSORBER_RADIUS = 0.04
SLOPE_ERROR_RAD = 2
SPEC_ERROR_RAD = 2.5

# 焦距精细扫描范围：0.8m 到 3.0m，步长 0.05m (共 45个点)
FOCAL_LENGTHS = np.linspace(0.8, 2.5, 35)

def update_config(width,focal_length,original_config):
    """读取并更新 config.json 文件，固定物理边界条件，并同步焦距与集热管坐标"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config = copy.deepcopy(original_config)
    scale_ratio = width / DEFALUT_WIDTH
    # 1. 设置固定的物理边界条件
    config['ParabolicTrough']['width'] = width
    config['ParabolicTrough']['slope_error'] = SLOPE_ERROR_RAD
    config['ParabolicTrough']['specularity_error'] = SPEC_ERROR_RAD
    config['Absorber']['r'] = ABSORBER_RADIUS
    
    # 2. 同步更新焦距与集热管中心高度
    config['ParabolicTrough']['focal_length'] = focal_length
    config['Absorber']['position'][2] = focal_length 

    # 3. 重新计算子镜bounds
    orig_bounds = original_config['ParabolicTrough']['bounds']
    half_w = width / 2.0
    center_offset = 0.075
    gap = 0.01
    
    # 提取原始左右两侧各子镜的纯镜面长度
    # 左侧镜面 (索引 0, 1, 2)
    w0 = orig_bounds[0][1] - orig_bounds[0][0]
    w1 = orig_bounds[1][1] - orig_bounds[1][0]
    w2 = orig_bounds[2][1] - orig_bounds[2][0]
    sum_L = w0 + w1 + w2
    
    # 右侧镜面 (索引 3, 4, 5)
    w3 = orig_bounds[3][1] - orig_bounds[3][0]
    w4 = orig_bounds[4][1] - orig_bounds[4][0]
    w5 = orig_bounds[5][1] - orig_bounds[5][0]
    sum_R = w3 + w4 + w5
    
    # 计算新开口下，单侧可用于排布纯镜面的可用总长度
    # 总长(half_w) - 中心留空(center_offset) - 两个子镜间隙(2 * gap)
    avail_mirrors = (half_w - center_offset) - (2 * gap)
    
    # 按原比例分配新的有效镜面宽度
    new_w0 = w0 / sum_L * avail_mirrors
    new_w1 = w1 / sum_L * avail_mirrors
    # w2 余量直接由坐标约束保证，无需单独分配
    
    new_w3 = w3 / sum_R * avail_mirrors
    new_w4 = w4 / sum_R * avail_mirrors
    
    # 依次拼接生成新的 bounds 坐标序列
    # --- 左侧 bounds (从外缘向中心收拢) ---
    L0_start = -half_w
    L0_end = L0_start + new_w0
    
    L1_start = L0_end + gap
    L1_end = L1_start + new_w1
    
    L2_start = L1_end + gap
    L2_end = -center_offset  # 强制锚定中心左边界
    
    # --- 右侧 bounds (从中心向外缘展开) ---
    R3_start = center_offset # 强制锚定中心右边界
    R3_end = R3_start + new_w3
    
    R4_start = R3_end + gap
    R4_end = R4_start + new_w4
    
    R5_start = R4_end + gap
    R5_end = half_w          # 强制锚定外缘
    
    config['ParabolicTrough']['bounds'] = [
        [round(L0_start, 5), round(L0_end, 5)],
        [round(L1_start, 5), round(L1_end, 5)],
        [round(L2_start, 5), round(L2_end, 5)],
        [round(R3_start, 5), round(R3_end, 5)],
        [round(R4_start, 5), round(R4_end, 5)],
        [round(R5_start, 5), round(R5_end, 5)]
    ]
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """运行程序并解析拦截率"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    
    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)', result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val * 100.0 if val <= 1.0 else val
    return 0.0

def main():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(EXEC_PATH):
        print("请检查 config.json 或可执行文件的路径。")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    try:
        results = {}  # 用于存储不同开口宽度下的结果
        
        print(f"开始多参数寻优... (开口范围: {APERTURE_WIDTH[0]:.1f}m - {APERTURE_WIDTH[-1]:.1f}m, 面型误差=2.0mrad)")
        
        # --- 双层循环扫描 ---
        for w in APERTURE_WIDTH:
            print(f"\n>> 正在测试开口宽度: {w:.2f}m")
            results[w] = {'intercepts': [], 'rim_angles': []}
            
            for fl in FOCAL_LENGTHS:
                update_config(w, fl,original_config)
                intercept = run_simulation()
                rim_angle_deg = np.degrees(2 * np.arctan(w / (4 * fl)))
                intercepted_power_per_meter = w * 1000.0 * (intercept / 100.0)
                
                results[w]['intercepts'].append(intercepted_power_per_meter)
                results[w]['rim_angles'].append(rim_angle_deg)
                # 可视化输出可按需取消注释
                print(f"  F={fl:.2f}m -> IF_power={intercepted_power_per_meter:.2f}%, Rim={rim_angle_deg:.1f}°")

        # --- 绘图逻辑 ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        # 颜色映射设置
        colors = cm.viridis(np.linspace(0, 0.9, len(APERTURE_WIDTH)))
        
        optimal_stats = []

        for i, w in enumerate(APERTURE_WIDTH):
            c = colors[i]
            intercepts = results[w]['intercepts']
            rim_angles = results[w]['rim_angles']
            
            # 记录并标注各开口宽度的最优解
            max_if = max(intercepts)
            opt_idx = intercepts.index(max_if)
            opt_fl = FOCAL_LENGTHS[opt_idx]
            opt_rim = rim_angles[opt_idx]
            optimal_stats.append(f"W={w:.1f}m: IF={max_if:.1f}%, F={opt_fl:.2f}m (Rim: {opt_rim:.1f}°)")
            
            # 绘制主轴（拦截率）：实线
            ax1.plot(FOCAL_LENGTHS, intercepts, marker='.', linestyle='-', color=c, 
                     linewidth=2, label=f'IF (W={w:.1f}m)')
            # 标注最优点
            ax1.plot(opt_fl, max_if, marker='*', color=c, markersize=12)
            
            # 绘制副轴（边缘角）：虚线，无标记
            ax2.plot(FOCAL_LENGTHS, rim_angles, linestyle='--', color=c, alpha=0.6)

        # 设置坐标轴
        ax1.set_xlabel('Focal Length (m)', fontsize=12)
        ax1.set_ylabel('Geometric Intercept Factor (%)', fontsize=12)
        ax2.set_ylabel('Rim Angle (degrees)', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.7)

        # 绘制固定的推荐边缘角区间 (在副轴 ax2 上)
        ax2.axhspan(85, 95, color='orange', alpha=0.15, label='Recommended Rim Angle (85°-95°)')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # 由于线条较多，将图例放在外部或分为两列
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10)

        plt.title('Intercept Factor & Rim Angle vs Focal Length (Varying Aperture Width)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 给底部图例留出空间
        
        plt.savefig('../out/multi_width_focal_length.png', dpi=300)
        print("\n✅ 寻优与绘图完成！图表已保存为 '../out/multi_width_focal_length.png'")
        print("\n焦距--拦截率极值参数汇总：")
        for stat in optimal_stats:
            print("  " + stat)
    
    finally:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(original_config, f, indent=4)
        print("\n🔄 已将 config.json 恢复至原始状态。")

if __name__ == "__main__":
    main()