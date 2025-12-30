#!/usr/bin/env python3
"""
PHASE1 Direction Change Detection - Colab Version with Visualization

Usage in Colab cell (ONE LINE):

# Example 1: Default parameters (lookback=5, confirm_local=True)
LOOKBACK = 5
CONFIRM_LOCAL = True
exec(open('https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_visualization.py').read())

# Or directly in cell:
LOOKBACK = 5
CONFIRM_LOCAL = True
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams

warnings.filterwarnings('ignore')

print("="*70)
print("PHASE1: Direction Change Detection with Visualization")
print("="*70)

print("\n[1/4] Loading data...")

try:
    from huggingface_hub import hf_hub_download
    csv_file = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_file)
    print("成功从HuggingFace下载")
except Exception as e:
    print(f"错误: {str(e)[:100]}")
    print("尝试使用本地文件...")
    import os
    if os.path.exists('labeled_klines_phase1.csv'):
        df = pd.read_csv('labeled_klines_phase1.csv')
        print("已从labeled_klines_phase1.csv加载")
    else:
        print("ERROR: 未找到数据文件")
        exit(1)

df['close'] = pd.to_numeric(df['close'])
df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

if 'timestamp' not in df.columns:
    if 'open_time' in df.columns:
        df['timestamp'] = df['open_time']
    else:
        df['timestamp'] = range(len(df))

print(f"数据大小: {df.shape}")
print(f"时间范围: {str(df['timestamp'].iloc[0])[:19]} 至 {str(df['timestamp'].iloc[-1])[:19]}")

# 使用Cell中的参数
try:
    lookback = LOOKBACK
    confirm_local = CONFIRM_LOCAL
except NameError:
    # 默认参数
    lookback = 5
    confirm_local = True

print(f"\n[2/4] 检测反转点 (lookback={lookback}, confirm_local={confirm_local})...")

reversals = []

def calculate_direction(idx, n):
    if idx < n:
        return 0
    return df.iloc[idx]['close'] - df.iloc[idx-n]['close']

def is_local_high(idx):
    if idx <= 0 or idx >= len(df) - 1:
        return False
    current = df.iloc[idx]['close']
    prev = df.iloc[idx-1]['close']
    next_val = df.iloc[idx+1]['close']
    return current >= prev and current >= next_val

def is_local_low(idx):
    if idx <= 0 or idx >= len(df) - 1:
        return False
    current = df.iloc[idx]['close']
    prev = df.iloc[idx-1]['close']
    next_val = df.iloc[idx+1]['close']
    return current <= prev and current <= next_val

for i in range(lookback + 1, len(df) - 1):
    prev_direction = calculate_direction(i-1, lookback)
    current_direction = calculate_direction(i, 1)
    
    if prev_direction > 0 and current_direction < 0:
        if confirm_local and not is_local_high(i-1):
            continue
        reversals.append({
            'index': i-1,
            'timestamp': df.iloc[i-1]['timestamp'],
            'close': df.iloc[i-1]['close'],
            'type': 'Local_High'
        })
    
    elif prev_direction < 0 and current_direction > 0:
        if confirm_local and not is_local_low(i-1):
            continue
        reversals.append({
            'index': i-1,
            'timestamp': df.iloc[i-1]['timestamp'],
            'close': df.iloc[i-1]['close'],
            'type': 'Local_Low'
        })

print(f"检测到 {len(reversals)} 个反转点")

print("\n[3/4] 标记数据集...")

labeled_df = df.copy()
labeled_df['Reversal_Label'] = 0
labeled_df['Reversal_Type'] = 'None'
labeled_df['Reversal_Price'] = 0.0

for reversal in reversals:
    i = int(reversal['index'])
    if i < len(labeled_df):
        labeled_df.at[i, 'Reversal_Label'] = 1
        labeled_df.at[i, 'Reversal_Type'] = reversal['type']
        labeled_df.at[i, 'Reversal_Price'] = reversal['close']

output_file = f'phase1_lb{lookback}_local{str(confirm_local)[0]}.csv'
cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
        'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
existing_cols = [c for c in cols if c in labeled_df.columns]
labeled_df[existing_cols].to_csv(output_file, index=False)

print(f"已保存至: {output_file}")

reversals_df = pd.DataFrame(reversals) if reversals else pd.DataFrame()
total = len(reversals)
local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low']) if total > 0 else 0
local_highs = len(reversals_df[reversals_df['type'] == 'Local_High']) if total > 0 else 0

print(f"\n{'='*70}")
print("PHASE1 Detection Results")
print(f"{'='*70}")

print(f"\n总反转点数: {total}")
print(f"  - 局部低点 (买入点): {local_lows}")
print(f"  - 局部高点 (卖出点): {local_highs}")
print(f"\n反转点占比: {total / len(df) * 100:.2f}%")
if total > 0:
    print(f"平均间距: {len(df) // (total + 1):.0f} 根K线")

if total > 0:
    print(f"\n前10个反转点:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].head(10).to_string())

print(f"\n{'='*70}")
print("[4/4] 生成可视化图表...")
print(f"{'='*70}")

if total > 0:
    rcParams['figure.figsize'] = (16, 7)
    
    # 使用最近800根K线
    start_idx = max(0, len(labeled_df) - 800)
    plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 绘制K线价格
    ax.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1.5, label='Close Price', zorder=1)
    
    # 标记Low点 (绿色上三角)
    lows = plot_df[plot_df['Reversal_Type'] == 'Local_Low']
    ax.scatter(lows.index, lows['close'], color='green', marker='^', s=150, label=f'Local Low ({len(lows)})', zorder=5, edgecolors='darkgreen', linewidth=1.5)
    
    # 标记High点 (红色下三角)
    highs = plot_df[plot_df['Reversal_Type'] == 'Local_High']
    ax.scatter(highs.index, highs['close'], color='red', marker='v', s=150, label=f'Local High ({len(highs)})', zorder=5, edgecolors='darkred', linewidth=1.5)
    
    ax.set_xlabel('K线索引', fontsize=12, fontweight='bold')
    ax.set_ylabel('价格 (USDT)', fontsize=12, fontweight='bold')
    ax.set_title(f'PHASE1 反转点检测 - lookback={lookback}, confirm_local={confirm_local}\nTotal: {len(plot_df[plot_df["Reversal_Label"]==1])} reversals in recent 800 bars', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(f'phase1_lb{lookback}_local{str(confirm_local)[0]}.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: phase1_lb{lookback}_local{str(confirm_local)[0]}.png")
    plt.show()
    
    print(f"\n最近800根K线中检测到 {len(plot_df[plot_df['Reversal_Label']==1])} 个反转点")
else:
    print("\n未检测到反转点")

print(f"\n{'='*70}")
print(f"PHASE1 Complete! 可以调整参数后重新运行")
print(f"{'='*70}")
