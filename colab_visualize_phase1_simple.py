#!/usr/bin/env python3
"""
PHASE1 Simple Reversal Visualization - Colab Version

Usage in Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_visualize_phase1_simple.py | python3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE1: Visualization of Simple Reversals")
print("="*70)

print("\nLoading data...")

try:
    df = pd.read_csv('labeled_klines_phase1_simple.csv')
    print("Loaded labeled_klines_phase1_simple.csv")
except:
    print("ERROR: labeled_klines_phase1_simple.csv not found")
    print("Please run PHASE1 first to generate the labeled data")
    exit(1)

df['close'] = pd.to_numeric(df['close'])
df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

print(f"Data shape: {df.shape}")
print(f"Total reversals: {df['Reversal_Label'].sum()}")

print("\nGenerating visualizations...")

def plot_section(df, start_idx, end_idx, section_num, total_sections):
    df_section = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    x = np.arange(len(df_section))
    width = 0.6
    
    for i in range(len(df_section)):
        row = df_section.iloc[i]
        o, c, h, l = row['open'], row['close'], row['high'], row['low']
        
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
    
    reversals = df_section[df_section['Reversal_Label'] == 1]
    
    local_lows = reversals[reversals['Reversal_Type'] == 'Local_Low']
    if len(local_lows) > 0:
        ax1.scatter(local_lows.index, local_lows['close'] - 100,
                   marker='^', color='green', s=300, alpha=0.8,
                   label='Local Low (Buy)', zorder=5,
                   edgecolors='darkgreen', linewidth=2)
    
    local_highs = reversals[reversals['Reversal_Type'] == 'Local_High']
    if len(local_highs) > 0:
        ax1.scatter(local_highs.index, local_highs['close'] + 100,
                   marker='v', color='red', s=300, alpha=0.8,
                   label='Local High (Sell)', zorder=5,
                   edgecolors='darkred', linewidth=2)
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'BTCUSDT 15min - Reversal Points (Section {section_num}/{total_sections})\n'
                  f'{df_section["timestamp"].iloc[0]} to {df_section["timestamp"].iloc[-1]}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, len(df_section))
    
    ax2.bar(x, df_section['volume'], color='steelblue', alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=10)
    ax2.set_xlabel('Index', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, len(df_section))
    
    plt.tight_layout()
    filename = f'reversal_section_{section_num}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.show()
    plt.close()

section_size = len(df) // 3

print("\n[1/3] Plotting section 1...")
plot_section(df, 0, section_size, 1, 3)

print("[2/3] Plotting section 2...")
plot_section(df, section_size, 2*section_size, 2, 3)

print("[3/3] Plotting section 3...")
plot_section(df, 2*section_size, len(df), 3, 3)

print("\n" + "="*70)
print("Visualization Complete!")
print("="*70)

reversals = df[df['Reversal_Label'] == 1]
local_lows = len(reversals[reversals['Reversal_Type'] == 'Local_Low'])
local_highs = len(reversals[reversals['Reversal_Type'] == 'Local_High'])

print(f"\nGenerated 3 charts with:")
print(f"  - Total reversal points: {len(reversals)}")
print(f"  - Local lows: {local_lows}")
print(f"  - Local highs: {local_highs}")
print(f"\nFiles: reversal_section_1.png, reversal_section_2.png, reversal_section_3.png")
