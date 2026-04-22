import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置参数 ---
CONFIG_PATH = '../resources/config.json'   
EXEC_PATH = '../build/mcrt_sim'          

# 焦距扫描范围：1.5m 到 3.0m，步长约 0.05m
FOCAL_LENGTHS = np.linspace(1.2, 2.7, 31)

# 挑选几个典型的面型误差水平进行对比 (单位: mrad)
SLOPE_ERRORS_MRAD = [0.0,1.5, 2.0, 2.5,3.0] 

# 固定管径为 DN80 进行测试
FIXED_RADIUS = 0.04

def update_config(focal_length, slope_error):
    """读取并同步更新焦距与集热管位置"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 1. 设置固定的管径和当前误差
    config['ParabolicTrough']['slope_error'] = slope_error
    
    # 2. 更新抛物面的焦距
    config['ParabolicTrough']['focal_length'] = focal_length
    
    # 3. 同步更新集热管的空间位置
    config['Absorber']['r'] = FIXED_RADIUS
    config['Absorber']['position'][2] = focal_length
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """运行程序并解析拦截率"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    
    # 根据实际输出调整正则表达式
    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)', result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val * 100.0 if val <= 1.0 else val
    return 0.0

def main():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(EXEC_PATH):
        print("请检查 config.json 或可执行文件的路径。")
        return

    # 用于存储不同误差下的曲线数据
    results = {error: [] for error in SLOPE_ERRORS_MRAD}

    print("开始进行 焦距-误差 敏感度扫描...")
    
    for error_mrad in SLOPE_ERRORS_MRAD:
        error_rad = error_mrad
        print(f"\n--- 当前面型误差: {error_mrad} mrad ---")
        
        for fl in FOCAL_LENGTHS:
            update_config(fl, error_rad)
            intercept = run_simulation()
            results[error_mrad].append(intercept)
            print(f"  焦距: {fl:.2f}m -> 拦截率: {intercept:.2f}%")

    # --- 绘制敏感度（曲率）对比图 ---
    plt.figure(figsize=(10, 6))
    
    colors = ['green', 'blue', 'yellow','red','orange']
    markers = ['o', 's', '^','D', 'v']
    
    for i, error_mrad in enumerate(SLOPE_ERRORS_MRAD):
        plt.plot(FOCAL_LENGTHS, results[error_mrad], 
                 marker=markers[i], linestyle='-', color=colors[i], 
                 label=f'Slope Error: {error_mrad} mrad')
    
    plt.title('Geometric Intercept Factor vs. Focal Length at Different Errors', fontsize=14)
    plt.xlabel('Focal Length (m)', fontsize=12)
    plt.ylabel('Geometric Intercept Factor (%)', fontsize=12)
    
    # 设置合理的Y轴范围以凸显曲率变化
    plt.ylim(60, 105) 
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.savefig('../out/focal_length_sensitivity.png', dpi=300)
    print("\n扫描完成！图表已保存为 '../out/focal_length_sensitivity.png'")
    # plt.show()

if __name__ == "__main__":
    main()