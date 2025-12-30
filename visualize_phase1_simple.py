#!/usr/bin/env python3
"""
PHASE1 Simple Reversal Visualization

Visualize the marked reversals on candlestick charts
with separate plots for different sections of data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def plot_reversals_section(df, start_idx, end_idx, section_num, total_sections):
    """
    Plot a section of candlestick chart with reversal markers.
    """
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
                  f'{df_section[\"timestamp\"].iloc[0]} to {df_section[\"timestamp\"].iloc[-1]}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)\n    ax1.set_xlim(-1, len(df_section))
    
    ax2.bar(x, df_section['volume'], color='steelblue', alpha=0.6, width=0.8)
    ax2.set_ylabel('Volume', fontsize=10)\n    ax2.set_xlabel('Index', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, len(df_section))
    
    plt.tight_layout()
    filename = f'reversal_section_{section_num}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()

def main():
    print(\"=\"*70)\n    print(\"PHASE1: Visualization of Simple Reversals\")\n    print(\"=\"*70)\n    \n    print(\"\\nLoading data...\")\n    df = pd.read_csv('labeled_klines_phase1_simple.csv')\n    \n    df['close'] = pd.to_numeric(df['close'])\n    df['open'] = pd.to_numeric(df['open'])\n    df['high'] = pd.to_numeric(df['high'])\n    df['low'] = pd.to_numeric(df['low'])\n    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)\n    \n    print(f\"Data shape: {df.shape}\")\n    print(f\"Total reversals: {df['Reversal_Label'].sum()}\")\n    \n    print(\"\\nGenerating visualizations...\")\n    \n    section_size = len(df) // 3\n    \n    print(\"\\n[1/3] Plotting section 1...\")\n    plot_reversals_section(df, 0, section_size, 1, 3)\n    \n    print(\"[2/3] Plotting section 2...\")\n    plot_reversals_section(df, section_size, 2*section_size, 2, 3)\n    \n    print(\"[3/3] Plotting section 3...\")\n    plot_reversals_section(df, 2*section_size, len(df), 3, 3)\n    \n    print(\"\\n\" + \"=\"*70)\n    print(\"Visualization Complete!\")\n    print(\"=\"*70)\n    \n    reversals = df[df['Reversal_Label'] == 1]\n    local_lows = len(reversals[reversals['Reversal_Type'] == 'Local_Low'])\n    local_highs = len(reversals[reversals['Reversal_Type'] == 'Local_High'])\n    \n    print(f\"\\nGenerated 3 charts with:\")\n    print(f\"  - Total reversal points: {len(reversals)}\")\n    print(f\"  - Local lows: {local_lows}\")\n    print(f\"  - Local highs: {local_highs}\")\n    print(f\"\\nFiles: reversal_section_1.png, reversal_section_2.png, reversal_section_3.png\")\n\nif __name__ == '__main__':\n    main()\n