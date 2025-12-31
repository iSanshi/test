import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 加载数据
file_path = './data/20251219_01/session.json'  # 请修改为您实际的文件路径
with open(file_path, 'r') as f:
    data = json.load(f)

# 2. 提取关键序列
iterations = range(1, data['max_iterations'] + 1)
info_gain = data['info_gain']
query_dist = data['query_distance']
# 推荐历史 (转换为 DataFrame 方便绘图)
# 注意：JSON中的推荐历史可能包含 null 或初始值，这里假设长度与迭代数一致或多一个
rec_history = np.array(data['recommendation_history'])
param_names = ['Intensity', 'Texture (Balance)', 'Rhythm (Speed)', 'Grain (Duty)']

# 3. 创建画布
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# --- 图表 A: 收敛性 (不确定性下降) ---
axes[0].plot(iterations, info_gain, marker='o', color='tab:blue', linewidth=2)
axes[0].set_title('Convergence: Information Gain (Uncertainty Reduction)', fontsize=14)
axes[0].set_ylabel('Info Gain (Entropy)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# --- 图表 B: 探索程度 (查询距离) ---
axes[1].bar(iterations, query_dist, color='tab:orange', alpha=0.7)
axes[1].axhline(y=1.414, color='r', linestyle='--', label='Diag of Unit Hypercube (approx)')
axes[1].set_title('Exploration Strategy: Query Distance', fontsize=14)
axes[1].set_ylabel('Euclidean Distance', fontsize=12)
axes[1].legend()
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

# --- 图表 C: 参数推荐轨迹 ---
for i in range(4):
    axes[2].plot(range(len(rec_history)), rec_history[:, i], label=param_names[i], linewidth=2)

axes[2].set_title('Optimization Trajectory: Best Guess Parameters', fontsize=14)
axes[2].set_ylabel('Parameter Value (20-100)', fontsize=12)
axes[2].set_xlabel('Iteration', fontsize=12)
axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[2].grid(True, linestyle='--', alpha=0.7)
axes[2].set_ylim(20, 100)

plt.tight_layout()
plt.show()