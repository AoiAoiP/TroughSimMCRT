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
# APERTURE_WIDTH = np.linspace(5.6, 9.1, 8)
APERTURE_WIDTH = 8.6
SLOPE_ERRORS_RAD = np.linspace(0.5, 3, 6)
# SLOPE_ERROR_RAD = 2
SPEC_ERROR_RAD = 2.5

# 焦距精细扫描范围：1.0m 到 3.0m，步长 0.05m (共 41个点)
FOCAL_LENGTHS = np.linspace(1.0, 3.0, 41)

def update_config(focal_length,slope_error,original_config):
    """读取并更新 config.json 文件，固定物理边界条件，并同步焦距与集热管坐标"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config = copy.deepcopy(original_config)
    # 1. 设置固定的物理边界条件
    config['ParabolicTrough']['slope_error'] = slope_error
    
    # 2. 同步更新焦距与集热管中心高度
    config['ParabolicTrough']['focal_length'] = focal_length
    config['Absorber']['position'][2] = focal_length 
    
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
        results = {}  # 用于存储不同开口斜率误差下的结果
        
        print(f"开始多参数寻优... (斜率误差范围: {SLOPE_ERRORS_RAD[0]:.1f}mrad - {SLOPE_ERRORS_RAD[-1]:.1f}mrad)")
        
        # --- 双层循环扫描 ---
        for s in SLOPE_ERRORS_RAD:
            print(f"\n>> 正在测试斜率误差: {s:.2f}mrad")
            results[s] = {'intercepts': [], 'rim_angles': []}
            
            for fl in FOCAL_LENGTHS:
                update_config(fl,s,original_config)
                intercept = run_simulation()
                rim_angle_deg = np.degrees(2 * np.arctan(APERTURE_WIDTH / (4 * fl)))
                # intercepted_power_per_meter = APERTURE_WIDTH * 1000.0 * (intercept / 100.0)
                
                results[s]['intercepts'].append(intercept)
                results[s]['rim_angles'].append(rim_angle_deg)
                # 可视化输出可按需取消注释
                print(f"  F={fl:.2f}mrad -> IF_power={intercept:.2f}%, Rim={rim_angle_deg:.1f}°")

        # --- 绘图逻辑 ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()
        
        # 颜色映射设置
        colors = cm.viridis(np.linspace(0, 0.9, len(SLOPE_ERRORS_RAD)))
        
        optimal_stats = []

        for i, s in enumerate(SLOPE_ERRORS_RAD):
            c = colors[i]
            intercepts = results[s]['intercepts']
            rim_angles = results[s]['rim_angles']
            
            # 记录并标注各开口宽度的最优解
            max_if = max(intercepts)
            opt_idx = intercepts.index(max_if)
            opt_fl = FOCAL_LENGTHS[opt_idx]
            opt_rim = rim_angles[opt_idx]
            optimal_stats.append(f"Slope={s:.1f}mrad: IF={max_if:.1f}%, F={opt_fl:.2f}m (Rim: {opt_rim:.1f}°)")
            
            # 绘制主轴（拦截率）：实线
            ax1.plot(FOCAL_LENGTHS, intercepts, marker='.', linestyle='-', color=c, 
                     linewidth=2, label=f'IF (Slope={s:.1f}mrad)')
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
        ax2.axhspan(80, 90, color='orange', alpha=0.15, label='Recommended Rim Angle (80°-90°)')

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # 由于线条较多，将图例放在外部或分为两列
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', 
                   bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=10)

        plt.title('Intercept Factor & Rim Angle vs Focal Length (Varying Slope Error)', fontsize=14)
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