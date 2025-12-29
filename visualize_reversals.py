#!/usr/bin/env python3
"""
Reversal Point Visualization Script

Generates comprehensive candlestick chart with marked reversal points
Usage:
  python visualize_reversals.py labeled_klines_phase1.csv
  or in Colab:
  %run visualize_reversals.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def create_reversal_chart(csv_file, start_idx=0, num_candles=500, figsize=(20, 10)):
    """
    Create candlestick chart with reversal points marked
    
    Args:
        csv_file: Path to labeled CSV file
        start_idx: Starting index for visualization
        num_candles: Number of candlesticks to display
        figsize: Figure size (width, height)
    """
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    end_idx = min(start_idx + num_candles, len(df))
    df_plot = df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print(f"Plotting {len(df_plot)} candlesticks ({start_idx} to {end_idx})")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # ============================================================
    # Main candlestick chart
    # ============================================================
    
    x = np.arange(len(df_plot))
    width = 0.6
    
    # Plot candlesticks
    for i in range(len(df_plot)):
        row = df_plot.iloc[i]
        o = row['open']
        c = row['close']
        h = row['high']
        l = row['low']
        
        # High-Low line (wick)
        ax1.plot([i, i], [l, h], 'k-', linewidth=1)
        
        # Open-Close body
        if c >= o:
            # Bullish candle (green)
            color = '#26a69a'
            height = c - o
            bottom = o
        else:
            # Bearish candle (red)
            color = '#ef5350'
            height = o - c
            bottom = c
        
        rect = Rectangle((i - width/2, bottom), width, height, 
                         facecolor=color, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
    
    # ============================================================
    # Mark reversal points
    # ============================================================
    
    reversals = df_plot[df_plot['Reversal_Label'] == 1]
    
    # Support reversals (green upward triangles)
    support = reversals[reversals['Reversal_Type'] == 'Support']
    if len(support) > 0:
        support_indices = support.index.tolist()
        support_prices = support['close'].values
        ax1.scatter(support_indices, support_prices - 100, marker='^', 
                   color='green', s=300, alpha=0.8, label='Support', zorder=5, edgecolors='darkgreen', linewidth=2)
    
    # Resistance reversals (red downward triangles)
    resistance = reversals[reversals['Reversal_Type'] == 'Resistance']
    if len(resistance) > 0:
        resistance_indices = resistance.index.tolist()
        resistance_prices = resistance['close'].values
        ax1.scatter(resistance_indices, resistance_prices + 100, marker='v', 
                   color='red', s=300, alpha=0.8, label='Resistance', zorder=5, edgecolors='darkred', linewidth=2)
    
    # Trend start up (blue upward arrows)
    trend_up = reversals[reversals['Reversal_Type'] == 'Trend_Start_Up']
    if len(trend_up) > 0:
        trend_up_indices = trend_up.index.tolist()
        trend_up_prices = trend_up['close'].values
        ax1.scatter(trend_up_indices, trend_up_prices - 150, marker='^', 
                   color='cyan', s=400, alpha=0.9, label='Trend Start Up', zorder=5, edgecolors='blue', linewidth=2)
    
    # Trend end down (orange downward arrows)
    trend_down = reversals[reversals['Reversal_Type'] == 'Trend_End_Down']
    if len(trend_down) > 0:
        trend_down_indices = trend_down.index.tolist()
        trend_down_prices = trend_down['close'].values
        ax1.scatter(trend_down_indices, trend_down_prices + 150, marker='v', 
                   color='orange', s=400, alpha=0.9, label='Trend End Down', zorder=5, edgecolors='darkorange', linewidth=2)
    
    # ============================================================
    # Plot moving averages
    # ============================================================
    
    if 'SMA20' in df_plot.columns:
        ax1.plot(x, df_plot['SMA20'], 'b-', alpha=0.5, linewidth=1.5, label='SMA20')
    if 'SMA50' in df_plot.columns:
        ax1.plot(x, df_plot['SMA50'], 'orange', alpha=0.5, linewidth=1.5, label='SMA50')
    if 'SMA200' in df_plot.columns:
        ax1.plot(x, df_plot['SMA200'], 'purple', alpha=0.5, linewidth=1.5, label='SMA200')
    
    # ============================================================
    # Plot Bollinger Bands
    # ============================================================
    
    if 'BB_Upper' in df_plot.columns and 'BB_Lower' in df_plot.columns:
        ax1.fill_between(x, df_plot['BB_Lower'], df_plot['BB_Upper'], 
                         alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(x, df_plot['BB_Upper'], 'gray', alpha=0.3, linewidth=0.8, linestyle='--')
        ax1.plot(x, df_plot['BB_Lower'], 'gray', alpha=0.3, linewidth=0.8, linestyle='--')
    
    # ============================================================
    # Bottom panel: RSI indicator
    # ============================================================
    
    if 'RSI' in df_plot.columns:
        ax2.plot(x, df_plot['RSI'], 'b-', linewidth=1.5, label='RSI')
        ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax2.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax2.fill_between(x, 30, 70, alpha=0.1, color='yellow')
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('RSI', fontsize=10)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # Styling
    # ============================================================
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'BTCUSDT 15min - Reversal Points Visualization\n'
                  f'{df_plot["timestamp"].iloc[0]} to {df_plot["timestamp"].iloc[-1]}',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, len(df_plot))
    
    # X-axis: show every Nth timestamp
    step = max(1, len(df_plot) // 20)
    ax1.set_xticks(range(0, len(df_plot), step))
    ax1.set_xticklabels([df_plot['timestamp'].iloc[i][:10] if i % (step*2) == 0 else '' 
                         for i in range(0, len(df_plot), step)], rotation=45, fontsize=8)
    
    ax2.set_xlabel('Time Index', fontsize=10)
    ax2.set_xlim(-1, len(df_plot))
    
    plt.tight_layout()
    
    # ============================================================
    # Save and show
    # ============================================================
    
    output_file = f'reversal_chart_{start_idx}_{end_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved to {output_file}")
    plt.show()
    
    # ============================================================
    # Print statistics
    # ============================================================
    
    print(f"\n{'='*70}")
    print("Reversal Statistics (shown range)")
    print(f"{'='*70}")
    print(f"Total candles: {len(df_plot)}")
    print(f"Total reversals: {len(reversals)}")
    print(f"  - Support: {len(support)}")
    print(f"  - Resistance: {len(resistance)}")
    print(f"  - Trend Start Up: {len(trend_up)}")
    print(f"  - Trend End Down: {len(trend_down)}")
    print(f"\nPrice range: {df_plot['close'].min():.2f} - {df_plot['close'].max():.2f}")
    print(f"RSI range: {df_plot['RSI'].min():.2f} - {df_plot['RSI'].max():.2f}")
    
    return fig, (ax1, ax2), reversals

def create_full_dataset_chart(csv_file, figsize=(20, 8)):
    """
    Create overview chart for entire dataset with sliding window
    """
    print(f"Loading full dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Downsample for overview
    step = max(1, len(df) // 1000)
    df_sample = df.iloc[::step].reset_index(drop=True)
    
    x = np.arange(len(df_sample))
    
    # Plot price
    ax.plot(x, df_sample['close'], 'k-', linewidth=1, label='Close Price')
    
    # Mark all reversal points
    reversals = df_sample[df_sample['Reversal_Label'] == 1]
    
    support = reversals[reversals['Reversal_Type'] == 'Support']
    if len(support) > 0:
        ax.scatter(support.index, support['close'], marker='^', 
                  color='green', s=100, alpha=0.7, label='Support')
    
    resistance = reversals[reversals['Reversal_Type'] == 'Resistance']
    if len(resistance) > 0:
        ax.scatter(resistance.index, resistance['close'], marker='v', 
                  color='red', s=100, alpha=0.7, label='Resistance')
    
    trend_up = reversals[reversals['Reversal_Type'] == 'Trend_Start_Up']
    if len(trend_up) > 0:
        ax.scatter(trend_up.index, trend_up['close'], marker='^', 
                  color='cyan', s=150, alpha=0.8, label='Trend Start Up', edgecolors='blue')
    
    trend_down = reversals[reversals['Reversal_Type'] == 'Trend_End_Down']
    if len(trend_down) > 0:
        ax.scatter(trend_down.index, trend_down['close'], marker='v', 
                  color='orange', s=150, alpha=0.8, label='Trend End Down', edgecolors='darkorange')
    
    ax.set_title(f'BTCUSDT 15min - Full Dataset Overview ({len(df)} candles)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Index (downsampled)', fontsize=11)
    ax.set_ylabel('Price (USDT)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'reversal_chart_overview.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Overview chart saved to {output_file}")
    plt.show()

if __name__ == '__main__':
    import sys
    
    csv_file = 'labeled_klines_phase1.csv' if len(sys.argv) < 2 else sys.argv[1]
    
    print("="*70)
    print("Reversal Point Visualization")
    print("="*70)
    
    # Create full overview
    print("\n[1/3] Creating full dataset overview...")
    create_full_dataset_chart(csv_file)
    
    # Create detailed charts for different sections
    df = pd.read_csv(csv_file)
    num_sections = 3
    section_size = len(df) // num_sections
    
    for section in range(num_sections):
        print(f"\n[{section+2}/{num_sections+1}] Creating detailed chart for section {section+1}...")
        start_idx = section * section_size
        create_reversal_chart(csv_file, start_idx=start_idx, num_candles=500)
    
    print("\n" + "="*70)
    print("All charts generated successfully!")
    print("="*70)
