import json
import subprocess
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- 配置路径 ---
CONFIG_PATH = '../resources/config.json'   
EXEC_PATH = '../build/mcrt_sim'          
SOLAR_DATA_TXT = '../resources/738_sundir_year.txt' # 你的太阳方向数据文件

# --- 待测管径字典 ---
TUBES = {
    'DN80': 0.040,  # 半径 40mm
    'DN90': 0.045   # 半径 45mm
}

def update_config_sun_dir(sun_vector, radius):
    """更新配置文件中的太阳方向和集热管半径"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 1. 更新集热管半径
    config['Absorber']['r'] = radius
    
    # 2. 更新太阳方向向量
    config['Sun']['dir'] = [sun_vector[0], sun_vector[1], sun_vector[2]]

    # 3. 拉长管长，避免末端效应
    config['ParabolicTrough']['length'] = 1000
    config['ParabolicTrough']['position'][2] = 500
    config['Absorber']['position'][2] = 500
    config['Absorber']['length'] = 1001
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def run_simulation():
    """运行 CUDA MCRT 程序并抓取几何拦截率"""
    result = subprocess.run([EXEC_PATH], capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    
    match = re.search(r'Intercept\s*Factor[:=]\s*([0-9.]+)', result.stdout, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        return val if val > 1.0 else val * 100.0  
    return 0.0

def main():
    if not os.path.exists(SOLAR_DATA_TXT):
        print(f"错误：找不到全年太阳数据文件 {SOLAR_DATA_TXT}")
        return

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    try:
        # 读取包含太阳向量的 CSV，假设列名为 'x', 'y', 'z', 以及可选的 'dni'
        df = pd.read_csv(SOLAR_DATA_TXT, sep=r'\s+', header=None, names=['x', 'y', 'z', 'unused'])
        
        # 如果没有提供 DNI，则默认所有有效时刻的 DNI 为常数 1000 W/m2 供加权使用
        if 'dni' not in df.columns:
            df['dni'] = 1000.0

        print(f"成功加载全年太阳数据，共 {len(df)} 个时刻。")
        
        annual_results = {}

        for tube_name, radius in TUBES.items():
            print(f"\n🚀 开始计算 {tube_name} (r={radius}m) 的全年能量加权拦截效率...")
            
            total_energy_hit = 0.0
            total_energy_available = 0.0
            valid_hours = 0
            
            start_time = time.time()

            for index, row in df.iterrows():
                x, z, y = row['x'], row['y'], row['z']
                dni = 1000
                    
                # 归一化向量，确保安全
                norm = np.sqrt(x**2 + y**2 + z**2)
                gx, gy, gz = x/norm, y/norm, z/norm
        
                # 假设槽的长度方向为 Y 轴。跟踪系统绕 Y 轴旋转，消除 X 轴方向的偏角。
                local_x = 0.0
                local_y = gy
                local_z = np.sqrt(gx**2 + gz**2)
                sun_vec = [local_x, local_y, local_z]
                cos_theta = local_z 
                
                # 更新 JSON 并跑仿真
                update_config_sun_dir(sun_vec, radius)
                intercept_factor_pct = run_simulation()
                
                # 能量加权累加 
                available_energy = dni * cos_theta
                hit_energy = available_energy * (intercept_factor_pct / 100.0)
                print(f"row:{valid_hours},intercept_factor_pct:{intercept_factor_pct:.2f}")
                
                total_energy_available += available_energy
                total_energy_hit += hit_energy
                valid_hours += 1
                
                # 进度提示
                if valid_hours % 500 == 0:
                    print(f"  已完成 {valid_hours} 个有效小时的计算...")

            # 计算该集热管的最终全年加权拦截率
            if total_energy_available > 0:
                annual_efficiency = (total_energy_hit / total_energy_available) * 100.0
            else:
                annual_efficiency = 0.0
                
            annual_results[tube_name] = annual_efficiency
            
            cost_time = time.time() - start_time
            print(f"✅ {tube_name} 计算完毕！耗时: {cost_time:.1f} 秒")
            print(f"📊 {tube_name} 全年能量加权拦截效率: {annual_efficiency:.2f}%")

        # --- 绘制对比柱状图供甲方汇报 ---
        plt.figure(figsize=(8, 6))
        bars = plt.bar(annual_results.keys(), annual_results.values(), color=['#1f77b4', '#ff7f0e'], width=0.5)
        
        # 添加数据标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('Annual Energy-Weighted Intercept Factor\n(DN80 vs DN90)', fontsize=14)
        plt.ylabel('Annual Intercept Efficiency (%)', fontsize=12)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        
        plt.savefig('annual_efficiency_comparison.png', dpi=300)
        print("\n结果图表已保存为 'annual_efficiency_comparison.png'")
        plt.show()
    finally:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(original_config, f, indent=4)
        print("\n🔄 已将 config.json 恢复至原始状态。")

if __name__ == "__main__":
    main()