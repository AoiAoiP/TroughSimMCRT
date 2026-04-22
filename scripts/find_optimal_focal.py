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
ABSORBER_RADIUS = 0.045
SLOPE_ERROR_RAD = 2
SPEC_ERROR_RAD = 2.5

# 焦距精细扫描范围：0.8m 到 3.0m，步长 0.05m (共 45个点)
FOCAL_LENGTHS = np.linspace(0.8, 3.0, 45)

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

    print(f"正在寻优... (开口={APERTURE_WIDTH}m, DN80, 面型误差=2.0mrad)")
    
    for fl in FOCAL_LENGTHS:
        update_config(fl)
        intercept = run_simulation()
        intercept_factors.append(intercept)
        print(f"  测试焦距: {fl:.2f}m -> 几何拦截率: {intercept:.2f}%")

    # --- 寻找最优焦距及其对应的拦截率 ---
    max_intercept = max(intercept_factors)
    optimal_idx = intercept_factors.index(max_intercept)
    optimal_fl = FOCAL_LENGTHS[optimal_idx]
    
    print("\n" + "="*40)
    print(f"✅ 寻优完成！")
    print(f"最优焦距取值: {optimal_fl:.2f} m")
    print(f"最大几何拦截率: {max_intercept:.2f}%")
    
    # 计算此时的边缘角 (Rim Angle)
    rim_angle_rad = 2 * np.arctan(APERTURE_WIDTH / (4 * optimal_fl))
    rim_angle_deg = np.degrees(rim_angle_rad)
    print(f"对应的系统边缘角: {rim_angle_deg:.1f}°")
    print("="*40 + "\n")

    # --- 绘制寻优曲线并标注极值点 ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(FOCAL_LENGTHS, intercept_factors, marker='o', linestyle='-', color='#2ca02c', linewidth=2)
    
    # 用红星标注最优点
    plt.plot(optimal_fl, max_intercept, marker='*', color='red', markersize=15, 
             label=f'Optimal F={optimal_fl:.2f}m\nMax IF={max_intercept:.2f}%')
    
    # 添加辅助线
    plt.axvline(x=optimal_fl, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=max_intercept, color='red', linestyle='--', alpha=0.5)
    
    plt.title(f'Optimal Focal Length Search\n(W={APERTURE_WIDTH}m, Tube=DN80, SlopeErr=2.0mrad)', fontsize=14)
    plt.xlabel('Focal Length (m)', fontsize=12)
    plt.ylabel('Geometric Intercept Factor (%)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='lower center')
    
    plt.tight_layout()
    plt.savefig('../out/optimal_focal_length.png', dpi=300)
    print("图表已保存为 '../out/optimal_focal_length.png'")
    plt.show()

if __name__ == "__main__":
    main()