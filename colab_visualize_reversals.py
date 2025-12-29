#!/usr/bin/env python3
"""
Colab Quick Visualization Script

Usage in Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_visualize_reversals.py | python3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("Reversal Point Visualization")
print("="*70)

print("\nLoading labeled data...")
df = pd.read_csv('labeled_klines_phase1.csv')

print(f"Data loaded: {len(df)} candles")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

num_sections = 3
section_size = len(df) // num_sections

for section in range(num_sections):
    start_idx = section * section_size
    end_idx = min(start_idx + 500, len(df))
    
    df_plot = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print(f"\n[Section {section+1}/{num_sections}] Plotting {len(df_plot)} candles...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    x = np.arange(len(df_plot))
    width = 0.6
    
    for i in range(len(df_plot)):
        row = df_plot.iloc[i]
        o = row['open']
        c = row['close']
        h = row['high']
        l = row['low']
        
        ax1.plot([i, i], [l, h], 'k-', linewidth=1)
        
        if c >= o:
            color = '#26a69a'
            height = c - o
            bottom = o
        else:
            color = '#ef5350'
            height = o - c
            bottom = c
        
        rect = Rectangle((i - width/2, bottom), width, height, 
                         facecolor=color, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
    
    reversals = df_plot[df_plot['Reversal_Label'] == 1]
    
    support = reversals[reversals['Reversal_Type'] == 'Support']
    if len(support) > 0:
        support_indices = support.index.tolist()
        support_prices = support['close'].values
        ax1.scatter(support_indices, support_prices - 100, marker='^', 
                   color='green', s=300, alpha=0.8, label='Support', zorder=5, edgecolors='darkgreen', linewidth=2)
    
    resistance = reversals[reversals['Reversal_Type'] == 'Resistance']
    if len(resistance) > 0:
        resistance_indices = resistance.index.tolist()
        resistance_prices = resistance['close'].values
        ax1.scatter(resistance_indices, resistance_prices + 100, marker='v', 
                   color='red', s=300, alpha=0.8, label='Resistance', zorder=5, edgecolors='darkred', linewidth=2)
    
    trend_up = reversals[reversals['Reversal_Type'] == 'Trend_Start_Up']
    if len(trend_up) > 0:
        trend_up_indices = trend_up.index.tolist()
        trend_up_prices = trend_up['close'].values
        ax1.scatter(trend_up_indices, trend_up_prices - 150, marker='^', 
                   color='cyan', s=400, alpha=0.9, label='Trend Start Up', zorder=5, edgecolors='blue', linewidth=2)
    
    trend_down = reversals[reversals['Reversal_Type'] == 'Trend_End_Down']
    if len(trend_down) > 0:
        trend_down_indices = trend_down.index.tolist()
        trend_down_prices = trend_down['close'].values
        ax1.scatter(trend_down_indices, trend_down_prices + 150, marker='v', 
                   color='orange', s=400, alpha=0.9, label='Trend End Down', zorder=5, edgecolors='darkorange', linewidth=2)
    
    if 'SMA20' in df_plot.columns:
        ax1.plot(x, df_plot['SMA20'], 'b-', alpha=0.5, linewidth=1.5, label='SMA20')
    if 'SMA50' in df_plot.columns:
        ax1.plot(x, df_plot['SMA50'], 'orange', alpha=0.5, linewidth=1.5, label='SMA50')
    if 'SMA200' in df_plot.columns:
        ax1.plot(x, df_plot['SMA200'], 'purple', alpha=0.5, linewidth=1.5, label='SMA200')
    
    if 'BB_Upper' in df_plot.columns and 'BB_Lower' in df_plot.columns:
        ax1.fill_between(x, df_plot['BB_Lower'], df_plot['BB_Upper'], 
                         alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(x, df_plot['BB_Upper'], 'gray', alpha=0.3, linewidth=0.8, linestyle='--')
        ax1.plot(x, df_plot['BB_Lower'], 'gray', alpha=0.3, linewidth=0.8, linestyle='--')
    
    if 'RSI' in df_plot.columns:
        ax2.plot(x, df_plot['RSI'], 'b-', linewidth=1.5, label='RSI')
        ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.fill_between(x, 30, 70, alpha=0.1, color='yellow')
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'BTCUSDT 15min - Reversal Points (Section {section+1}/{num_sections})\n'
                  f'{df_plot["timestamp"].iloc[0]} to {df_plot["timestamp"].iloc[-1]}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, len(df_plot))
    
    step = max(1, len(df_plot) // 20)
    ax1.set_xticks(range(0, len(df_plot), step))
    ax1.set_xticklabels([df_plot['timestamp'].iloc[i][:16] if i % (step*2) == 0 else '' 
                         for i in range(0, len(df_plot), step)], rotation=45, fontsize=8)
    
    ax2.set_xlabel('Index', fontsize=10)
    ax2.set_xlim(-1, len(df_plot))
    
    plt.tight_layout()
    plt.savefig(f'reversal_chart_section{section+1}.png', dpi=100, bbox_inches='tight')
    print(f"Saved: reversal_chart_section{section+1}.png")
    plt.show()
    
    print(f"\nSection {section+1} Statistics:")
    print(f"  Total reversals: {len(reversals)}")
    print(f"  - Support: {len(support)}")
    print(f"  - Resistance: {len(resistance)}")
    print(f"  - Trend Start Up: {len(trend_up)}")
    print(f"  - Trend End Down: {len(trend_down)}")

print("\n" + "="*70)
print("Visualization Complete!")
print("="*70)
print(f"\nTotal Statistics:")
print(f"  Total candles: {len(df)}")
print(f"  Total reversals: {(df['Reversal_Label'] == 1).sum()}")
print(f"  Support: {(df['Reversal_Type'] == 'Support').sum()}")
print(f"  Resistance: {(df['Reversal_Type'] == 'Resistance').sum()}")
print(f"  Trend Start Up: {(df['Reversal_Type'] == 'Trend_Start_Up').sum()}")
print(f"  Trend End Down: {(df['Reversal_Type'] == 'Trend_End_Down').sum()}")
