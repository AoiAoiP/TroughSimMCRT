import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 核心工程物理参数配置 ---
CONFIG_PATH = '../resources/config.json'   
EXEC_PATH = '../build/mcrt_sim'          

# 槽式开口宽度
APERTURE_WIDTH = 8.61502 
ABSORBER_RADIUS = 0.04
SLOPE_ERROR_RAD = 2
SPEC_ERROR_RAD = 2.5

# 焦距精细扫描范围：0.8m 到 3.0m，步长 0.05m (共 45个点)
FOCAL_LENGTHS = np.linspace(0.8, 2.5, 35)

def update_config(focal_length):
    """读取并更新 config.json 文件，固定物理边界条件，并同步焦距与集热管坐标"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 1. 设置固定的物理边界条件
    config['ParabolicTrough']['width'] = APERTURE_WIDTH
    config['ParabolicTrough']['slope_error'] = SLOPE_ERROR_RAD
    config['ParabolicTrough']['specularity_error'] = SPEC_ERROR_RAD
    config['Absorber']['r'] = ABSORBER_RADIUS
    
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

    intercept_factors = []
    rim_angles_deg = []

    print(f"正在寻优... (开口={APERTURE_WIDTH}m, DN80, 面型误差=2.0mrad)")
    
    for fl in FOCAL_LENGTHS:
        update_config(fl)
        intercept = run_simulation()
        rim_angle_deg = np.degrees(2 * np.arctan(APERTURE_WIDTH / (4 * fl)))
        intercept_factors.append(intercept)
        rim_angles_deg.append(rim_angle_deg)
        print(f"  测试焦距: {fl:.2f}m -> 几何拦截率: {intercept:.2f}%, 边缘角：{rim_angle_deg:.2f}°")

    # --- 寻找最优焦距及其对应的拦截率 ---
    max_intercept = max(intercept_factors)
    optimal_idx = intercept_factors.index(max_intercept)
    optimal_fl = FOCAL_LENGTHS[optimal_idx]
    
    print("\n" + "="*40)
    print(f"✅ 寻优完成！")
    print(f"最优焦距取值: {optimal_fl:.2f} m")
    print(f"最大几何拦截率: {max_intercept:.2f}%")
    rim_angle = np.degrees(2 * np.arctan(APERTURE_WIDTH / (4 * optimal_fl)))
    print(f"对应的系统边缘角: {rim_angle:.2f}°")
    print("="*40 + "\n")

    # --- 计算推荐区间 (85°-95°) 对应的焦距 ---
    def phi_to_f(phi_deg):
        """边缘角转焦距公式的逆运算"""
        return APERTURE_WIDTH / (4 * np.tan(np.radians(phi_deg / 2)))

    f_start = phi_to_f(95)  # 边缘角大，焦距小
    f_end = phi_to_f(85)    # 边缘角小，焦距大
    
    # 筛选推荐区间内的数据点进行分析
    rec_mask = (FOCAL_LENGTHS >= f_start) & (FOCAL_LENGTHS <= f_end)
    rec_focals = FOCAL_LENGTHS[rec_mask]
    rec_intercepts = np.array(intercept_factors)[rec_mask]
    
    if len(rec_intercepts) > 0:
        avg_rec_if = np.mean(rec_intercepts)
        max_rec_if = np.max(rec_intercepts)
    else:
        avg_rec_if = max_rec_if = 0.0

    print(f" 工程推荐区间分析 (85°~95°):")
    print(f"  对应焦距范围: {f_start:.3f}m - {f_end:.3f}m")
    print(f"  区间内最大拦截率: {max_rec_if:.2f}%")

    # --- 绘制寻优曲线并标注极值点 ---
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # 绘制主轴：几何拦截率 (Green)
    line1 = ax1.plot(FOCAL_LENGTHS, intercept_factors, marker='o', linestyle='-', 
                     color='#2ca02c', linewidth=2, label='Intercept Factor (%)')
    ax1.set_xlabel('Focal Length (m)', fontsize=12)
    ax1.set_ylabel('Geometric Intercept Factor (%)', fontsize=12, color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # 用红星标注最优点
    ax1.plot(optimal_fl, max_intercept, marker='*', color='red', markersize=15, 
             label=f'Optimal: F={optimal_fl:.2f}m, IF={max_intercept:.2f}%')
    ax1.axvline(x=optimal_fl, color='red', linestyle='--', alpha=0.3)

    # 创建边缘角副轴
    ax2 = ax1.twinx()
    line2 = ax2.plot(FOCAL_LENGTHS, rim_angles_deg, linestyle='--', color='#1f77b4', 
                     linewidth=2, label='Rim Angle (deg)')
    ax2.set_ylabel('Rim Angle (degrees)', fontsize=12, color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # --- 在 ax2 (边缘角轴) 上绘制推荐区间阴影 ---
    ax2.axhspan(85, 95, color='orange', alpha=0.2, label='Recommended Rim Angle (85°-95°)')
    
    # 在 ax1 (焦距轴) 上同步绘制竖向填充，方便对齐
    ax1.axvspan(f_start, f_end, color='gray', alpha=0.1, linestyle='--')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)

    plt.title(f'Optimal Focal Length & Rim Angle\n(Aperture={APERTURE_WIDTH}m, DN80, SlopeErr=2.0mrad)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('../out/optimal_focal_length.png', dpi=300)
    print("图表已保存为 '../out/optimal_focal_length.png'")
    # plt.show()

if __name__ == "__main__":
    main()