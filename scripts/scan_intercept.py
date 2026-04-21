import json
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置参数 ---
CONFIG_PATH = '../resources/config.json'   # 根据你的实际路径调整
EXEC_PATH = '../build/mcrt_sim'          # CUDA 程序的执行路径
SLOPE_ERRORS_MRAD = np.linspace(0, 5, 11)  # 扫描范围：0 到 5 mrad，共 11 个点
TUBES = {
    'DN80': 0.040,  # 半径 40mm
    'DN90': 0.045   # 半径 45mm
}

def update_config(radius, slope_error):
    """读取并更新 config.json 文件"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config['Absorber']['r'] = radius
    config['ParabolicTrough']['slope_error'] = slope_error
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """运行 CUDA 程序并解析标准输出中的拦截率"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"CUDA 程序运行失败:\n{result.stderr}")
        return 0.0
    
    output = result.stdout
    
    # --- 核心：正则表达式提取 ---
    # CUDA软件运行日志  "Intercept Factor: 0.985"
    # 需要根据实际终端输出修改此正则！
    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)', output, re.IGNORECASE)
    
    if match:
        intercept_factor = float(match.group(1))
        if intercept_factor <= 1.0:
            intercept_factor *= 100.0 
        return intercept_factor
    else:
        print("未在输出中找到拦截率数据，请检查正则表达式或程序输出。")
        print("实际输出为:", output[-200:]) # 打印最后 200 个字符帮助调试
        return 0.0

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"找不到配置文件: {CONFIG_PATH}")
        return
    if not os.path.exists(EXEC_PATH):
        print(f"找不到可执行文件: {EXEC_PATH}，请先编译 C++ 项目。")
        return

    results = {name: [] for name in TUBES.keys()}

    print("开始自动化 MCRT 仿真扫描...")
    
    for name, radius in TUBES.items():
        print(f"\n--- 开始模拟集热管: {name} (r={radius}m) ---")
        for error in SLOPE_ERRORS_MRAD:
            error_rad = error  # 转换为弧度送入仿真
            
            # 1. 更新配置
            update_config(radius, error_rad)
            
            # 2. 运行并抓取结果
            intercept = run_simulation()
            results[name].append(intercept)
            
            print(f"面型误差: {error:.1f} mrad | 拦截率: {intercept:.2f}%")

    # --- 绘制对比曲线 ---
    plt.figure(figsize=(10, 6))
    
    # 绘制 DN80 和 DN90 曲线
    plt.plot(SLOPE_ERRORS_MRAD, results['DN80'], marker='o', linestyle='-', color='b', label='DN80 (r=40mm)')
    plt.plot(SLOPE_ERRORS_MRAD, results['DN90'], marker='s', linestyle='--', color='r', label='DN90 (r=45mm)')
    
    plt.title('Geometric Intercept Factor vs. Slope Error', fontsize=14)
    plt.xlabel('Slope Error (mrad)', fontsize=12)
    plt.ylabel('Geometric Intercept Factor (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存并显示图像
    plt.savefig('intercept_factor_comparison.png', dpi=300)
    print("\n扫描完成！图表已保存为 'intercept_factor_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()