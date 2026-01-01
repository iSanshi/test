"""
verify_uncertainty.py
用于验证当前代码库中 Uncertainty Level (1-5) 映射是否反转的对比测试脚本。

原理：
我们建立两个完全一样的 GP 模型。
- 场景 A：用户非常确定 (Level 5)，选择 A > B。
- 场景 B：用户非常不确定 (Level 1)，选择 A > B。

预期（如果逻辑正确）：场景 A 的模型应该学到更多，A 点的得分应该变得更高。
预期（如果逻辑反了）：场景 B 的模型得分反而更高，因为 Sigma 小。
"""

import numpy as np
import sys
import os

# 确保能导入 src 模块
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preference_learning.interface.session import PreferenceSession, SessionMode

def run_simulation(level_to_test):
    """
    模拟一次单轮交互，返回模型对胜者的预测评分
    """
    # 1. 初始化 Session
    session = PreferenceSession()
    session.start(mode=SessionMode.USER, max_iterations=5, gt_label="Gaussian (center)")
    
    # 2. 构造两个模拟的候选点 (归一化空间 [0,1])
    # P1: 靠近中心 (我们让它赢)
    # P2: 靠近边缘
    p1_norm = np.array([0.5, 0.5, 0.5, 0.5]) 
    p2_norm = np.array([0.1, 0.1, 0.1, 0.1])
    
    p1_phys = session.gp.denormalize_parameters(p1_norm)
    p2_phys = session.gp.denormalize_parameters(p2_norm)
    
    # 3. 手动注入 current_candidates (模拟 UI 生成了 query)
    # 我们这里不做复杂的 AudioCandidate 包装，直接调用底层 GP 更新
    # 模拟 session.record_user_choice 的核心逻辑
    
    y = 1 # Choice A (P1 wins)
    
    # --- 核心关注点：这里就是 session.py 里当前的代码逻辑 ---
    # session.py 原文: self.gp.update_parameters([p1_norm, p2_norm], y, level, self.pref_dict)
    # 我们完全复刻这个调用
    session.gp.update_parameters([p1_norm, p2_norm], y, level_to_test, session.pref_dict)
    
    # 4. 看看模型学得怎么样
    # 计算 P1 (胜者) 在更新后的后验均值 (Posterior Mean)
    # 初始均值是 0.0。学得越确信，这个值应该越高 (越接近 1 或更高)。
    mean_val = session.gp.mean1pt(p1_norm)
    
    # 这里必须处理 float/array 的返回类型差异
    if isinstance(mean_val, (list, tuple, np.ndarray)):
        score = float(mean_val[0])
    else:
        score = float(mean_val)
        
    return score

def main():
    print("="*60)
    print("Uncertainty Mapping Verification Test")
    print("="*60)
    
    # 测试 Level 1 (UI定义: 非常不确定 / Low Confidence)
    score_lvl_1 = run_simulation(level_to_test=1)
    
    # 测试 Level 5 (UI定义: 非常确定 / High Confidence)
    score_lvl_5 = run_simulation(level_to_test=5)
    
    print(f"场景 A [Level 1 - 用户说'不确定']: 模型打分 = {score_lvl_1:.4f}")
    print(f"场景 B [Level 5 - 用户说'很确定']: 模型打分 = {score_lvl_5:.4f}")
    
    print("-" * 60)
    print("结果分析:")
    
    diff = score_lvl_1 - score_lvl_5
    
    if score_lvl_1 > score_lvl_5:
        print("🔴 结论: 逻辑反了 (BUG CONFIRMED)")
        print(f"    Level 1 (本该忽略) 导致模型权重增加了 {score_lvl_1:.4f}")
        print(f"    Level 5 (本该重视) 导致模型权重仅增加 {score_lvl_5:.4f}")
        print("    原因: GP 内部 Sigma 定义是 1->0.01(强), 5->9.0(弱)，但 Session 直接透传了 Level。")
    elif score_lvl_5 > score_lvl_1:
        print("🟢 结论: 逻辑正确")
        print("    Level 5 产生了更强的模型更新。")
    else:
        print("🟡 结论: 无区别 (可能模型配置有误)")

if __name__ == "__main__":
    main()