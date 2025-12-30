#!/usr/bin/env python3
"""
PHASE1 Hybrid Labeling System - Trend Reversals + Sideways Support/Resistance

Usage in Colab:

!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_hybrid_labeling.py -O phase1_hybrid.py

# 趣势阶段 + 横盘民量区
!python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode hybrid

# 仅趣势
!python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode trend

# 仅横盘
!python3 phase1_hybrid.py --lookback 7 --detect_mode sideways

Logic:
  1. 横盘判断: 最高最低的比率 < 阻值 (e.g., 0.5%)
  2. 趣势阶段: 横盘判断 = False
  3. 横盘区: 找最高/最低作为压力/支撑
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import sys
from scipy import signal

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description='PHASE1 Hybrid Labeling - Trend + Sideways',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode hybrid
  python3 phase1_hybrid.py --lookback 7 --detect_mode trend
  python3 phase1_hybrid.py --detect_mode sideways
        """
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=7,
        help='向后看N根K线 (default: 7)'
    )
    
    parser.add_argument(
        '--amplitude_threshold',
        type=float,
        default=0.656,
        help='趣势涨跌幅阈值(%) (default: 0.656)'
    )
    
    parser.add_argument(
        '--sideways_threshold',
        type=float,
        default=0.5,
        help='横盘判断阈值: 最高最低比率 < X% (default: 0.5)'
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        default=30,
        help='横盘检测窗口 (default: 30)'
    )
    
    parser.add_argument(
        '--detect_mode',
        type=str,
        default='hybrid',
        choices=['hybrid', 'trend', 'sideways'],
        help='检测模式 (default: hybrid)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件名'
    )
    
    return parser.parse_args()

def detect_sideways_segments(df, window_size, threshold):
    """
    检测横盘段
    横盘条件: 一个window内最高最低比率 < threshold%
    """
    sideways_segments = []
    
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        high = window['high'].max()
        low = window['low'].min()
        ratio = ((high - low) / low) * 100
        
        if ratio < threshold:
            # 找该段的最高和最低
            resistance = high
            support = low
            segment = {
                'start_idx': i,
                'end_idx': i + window_size - 1,
                'resistance': resistance,
                'support': support,
                'range': high - low,
                'range_percent': ratio
            }
            sideways_segments.append(segment)
    
    # 合并相邻的横盘段
    if not sideways_segments:
        return []
    
    merged = []
    current = sideways_segments[0].copy()
    
    for seg in sideways_segments[1:]:
        # 如果有重叠或接近，就合并
        if seg['start_idx'] <= current['end_idx'] + 5:
            current['end_idx'] = max(current['end_idx'], seg['end_idx'])
            current['resistance'] = max(current['resistance'], seg['resistance'])
            current['support'] = min(current['support'], seg['support'])
            current['range'] = current['resistance'] - current['support']
            current['range_percent'] = ((current['resistance'] - current['support']) / current['support']) * 100
        else:
            merged.append(current)
            current = seg.copy()
    
    merged.append(current)
    return merged

def detect_trend_reversals(df, lookback, amplitude_threshold, lookforward):
    """
    检测趣势段的反转点
    """
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
    
    def check_amplitude(idx, direction_type, threshold, look_ahead):
        reversal_price = df.iloc[idx]['close']
        max_idx = min(idx + look_ahead, len(df) - 1)
        
        if direction_type == 'up':
            max_price = df.iloc[idx:max_idx+1]['high'].max()
            amplitude = ((max_price - reversal_price) / reversal_price) * 100
        else:
            min_price = df.iloc[idx:max_idx+1]['low'].min()
            amplitude = ((reversal_price - min_price) / reversal_price) * 100
        
        return amplitude, amplitude >= threshold
    
    for i in range(lookback + 1, len(df) - lookforward):
        prev_direction = calculate_direction(i-1, lookback)
        current_direction = calculate_direction(i, 1)
        
        if prev_direction > 0 and current_direction < 0:
            if not is_local_high(i-1):
                continue
            amplitude, is_valid = check_amplitude(i-1, 'down', amplitude_threshold, lookforward)
            if is_valid:
                reversals.append({
                    'index': i-1,
                    'type': 'Local_High',
                    'amplitude': amplitude
                })
        
        elif prev_direction < 0 and current_direction > 0:
            if not is_local_low(i-1):
                continue
            amplitude, is_valid = check_amplitude(i-1, 'up', amplitude_threshold, lookforward)
            if is_valid:
                reversals.append({
                    'index': i-1,
                    'type': 'Local_Low',
                    'amplitude': amplitude
                })
    
    return reversals

def main():
    args = parse_args()
    
    lookback = args.lookback
    amplitude_threshold = args.amplitude_threshold
    sideways_threshold = args.sideways_threshold
    window_size = args.window_size
    detect_mode = args.detect_mode
    lookforward = 10
    
    print("="*70)
    print("PHASE1: Hybrid Labeling - Trend Reversals + Sideways Support/Resistance")
    print("="*70)
    
    print(f"\n[参数设置]")
    print(f"  --lookback {lookback}")
    print(f"  --amplitude_threshold {amplitude_threshold}%")
    print(f"  --sideways_threshold {sideways_threshold}%")
    print(f"  --window_size {window_size}")
    print(f"  --detect_mode {detect_mode}")
    
    print("\n[1/4] Loading data...")
    
    try:
        from huggingface_hub import hf_hub_download
        csv_file = hf_hub_download(
            repo_id="zongowo111/cpb-models",
            filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
            repo_type="dataset"
        )
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        sys.exit(1)
    
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
    
    print(f"Data size: {df.shape}")
    print(f"Time range: {str(df['timestamp'].iloc[0])[:19]} to {str(df['timestamp'].iloc[-1])[:19]}")
    
    # 初始化标记
    labeled_df = df.copy()
    labeled_df['Market_Mode'] = 'Unknown'  # Trend or Sideways
    labeled_df['Label_Type'] = 'None'  # Reversal, Resistance, Support
    labeled_df['Label_Value'] = 0.0
    labeled_df['Label_Strength'] = 0.0  # amplitude or range
    
    print("\n[2/4] Detecting market segments...")
    
    # 检测横盘段
    sideways_segments = detect_sideways_segments(df, window_size, sideways_threshold)
    print(f"Detected {len(sideways_segments)} sideways segments")
    
    # 一开始，所有都是Trend
    labeled_df['Market_Mode'] = 'Trend'
    
    # 控制横盘段
    for seg in sideways_segments:
        start = int(seg['start_idx'])
        end = int(seg['end_idx']) + 1
        labeled_df.loc[start:end, 'Market_Mode'] = 'Sideways'
    
    trend_count = (labeled_df['Market_Mode'] == 'Trend').sum()
    sideways_count = (labeled_df['Market_Mode'] == 'Sideways').sum()
    print(f"Trend bars: {trend_count}, Sideways bars: {sideways_count}")
    print(f"Trend ratio: {trend_count/len(df)*100:.1f}%, Sideways ratio: {sideways_count/len(df)*100:.1f}%")
    
    trend_reversals = 0
    resistance_marks = 0
    support_marks = 0
    
    print("\n[3/4] Labeling...")
    
    if detect_mode in ['trend', 'hybrid']:
        print("  Detecting trend reversals...")
        reversals = detect_trend_reversals(df, lookback, amplitude_threshold, lookforward)
        
        for reversal in reversals:
            idx = int(reversal['index'])
            if idx < len(labeled_df):
                labeled_df.at[idx, 'Label_Type'] = reversal['type']
                labeled_df.at[idx, 'Label_Value'] = df.iloc[idx]['close']
                labeled_df.at[idx, 'Label_Strength'] = reversal['amplitude']
                trend_reversals += 1
        
        print(f"  Found {trend_reversals} trend reversals")
    
    if detect_mode in ['sideways', 'hybrid']:
        print("  Marking sideways support/resistance...")
        
        for seg in sideways_segments:
            start = int(seg['start_idx'])
            end = int(seg['end_idx']) + 1
            
            # 标记第一个阻力
            idx = start
            labeled_df.at[idx, 'Label_Type'] = 'Resistance'
            labeled_df.at[idx, 'Label_Value'] = seg['resistance']
            labeled_df.at[idx, 'Label_Strength'] = seg['range_percent']
            resistance_marks += 1
            
            # 标记第一个支撇
            labeled_df.at[idx+5 if idx+5 < end else idx, 'Label_Type'] = 'Support'
            labeled_df.at[idx+5 if idx+5 < end else idx, 'Label_Value'] = seg['support']
            labeled_df.at[idx+5 if idx+5 < end else idx, 'Label_Strength'] = seg['range_percent']
            support_marks += 1
        
        print(f"  Marked {resistance_marks} resistance + {support_marks} support zones")
    
    # 保存CSV
    if args.output:
        output_file = args.output
    else:
        output_file = f'phase1_hybrid_lb{lookback}_amp{amplitude_threshold}_{detect_mode}.csv'
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Market_Mode', 'Label_Type', 'Label_Value', 'Label_Strength']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    labeled_df[existing_cols].to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("PHASE1 Hybrid Labeling Results")
    print(f"{'='*70}")
    print(f"\nMarket Segmentation:")
    print(f"  Trend bars: {trend_count} ({trend_count/len(df)*100:.1f}%)")
    print(f"  Sideways bars: {sideways_count} ({sideways_count/len(df)*100:.1f}%)")
    
    if detect_mode in ['trend', 'hybrid']:
        print(f"\nTrend Reversals: {trend_reversals}")
    if detect_mode in ['sideways', 'hybrid']:
        print(f"\nSideways Zones:")
        print(f"  Resistance marks: {resistance_marks}")
        print(f"  Support marks: {support_marks}")
    
    print(f"\n{'='*70}")
    print("[4/4] Generating visualization...")
    print(f"{'='*70}")
    
    # 可视化
    rcParams['figure.figsize'] = (16, 7)
    start_idx = max(0, len(labeled_df) - 800)
    plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # K线圖
    ax1.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1.5, label='Close Price', zorder=1)
    
    # 横盘检测
    sideways_mask = plot_df['Market_Mode'] == 'Sideways'
    ax1.fill_between(range(len(plot_df)), plot_df['close'].min()-100, plot_df['close'].max()+100,
                      where=sideways_mask, alpha=0.1, color='blue', label='Sideways', zorder=0)
    
    # 趣势反转点
    if detect_mode in ['trend', 'hybrid']:
        highs = plot_df[(plot_df['Label_Type'] == 'Local_High')]
        lows = plot_df[(plot_df['Label_Type'] == 'Local_Low')]
        ax1.scatter(highs.index, highs['close'], color='red', marker='v', s=150, label=f'Local High ({len(highs)})', zorder=5)
        ax1.scatter(lows.index, lows['close'], color='green', marker='^', s=150, label=f'Local Low ({len(lows)})', zorder=5)
    
    # 横盘支撇/阻力
    if detect_mode in ['sideways', 'hybrid']:
        resist = plot_df[(plot_df['Label_Type'] == 'Resistance')]
        support = plot_df[(plot_df['Label_Type'] == 'Support')]
        ax1.scatter(resist.index, resist['close'], color='orange', marker='s', s=120, label=f'Resistance ({len(resist)})', zorder=5)
        ax1.scatter(support.index, support['close'], color='purple', marker='s', s=120, label=f'Support ({len(support)})', zorder=5)
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PHASE1 Hybrid Labeling - Mode: {detect_mode}\nTrend: {trend_count} | Sideways: {sideways_count}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 市场模式下方
    mode_colors = plot_df['Market_Mode'].map({'Trend': 0, 'Sideways': 1})
    ax2.fill_between(range(len(plot_df)), 0, mode_colors, alpha=0.5, color='steelblue')
    ax2.set_ylabel('Market Mode', fontsize=12, fontweight='bold')
    ax2.set_xlabel('K-line Index', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Trend', 'Sideways'])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    png_file = output_file.replace('.csv', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {png_file}")
    plt.show()
    
    print(f"\n{'='*70}")
    print("PHASE1 Hybrid Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
