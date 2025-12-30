#!/usr/bin/env python3
"""
PHASE1 Range Zone Detection - Improved Sideways Detection with Better Visualization

Usage in Colab:

!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_range_detection.py -O phase1_range.py

# 例1: 横盘区间检测 (阈值 0.8%)
!python3 phase1_range.py --lookback 8 --amplitude_threshold 0.65 --range_threshold 0.8 --min_range_bars 20 --detect_mode hybrid

# 例2: 更严格的区间 (阈值 1.0%)
!python3 phase1_range.py --range_threshold 1.0 --min_range_bars 30

# 例3: 更敏感的区间 (阈值 0.5%)
!python3 phase1_range.py --range_threshold 0.5 --min_range_bars 15

range_threshold: 区间判断阈值 (%)
  - 越大 = 更难判定为区间，但段数较少
  - 越小 = 更容易判定为区间，但幅度式变区可能被分解

min_range_bars: 最低区间长度 (根数)
  - 小段摣不置了。最少㖠10-20根。
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import sys

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(
        description='PHASE1 Range Zone Detection with Improved Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 phase1_range.py --lookback 8 --amplitude_threshold 0.65 --range_threshold 0.8
  python3 phase1_range.py --range_threshold 1.0 --min_range_bars 30
  python3 phase1_range.py --range_threshold 0.5 --min_range_bars 15
        """
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=8,
        help='趋势判断周期 (default: 8)'
    )
    
    parser.add_argument(
        '--amplitude_threshold',
        type=float,
        default=0.65,
        help='趋势反转最低幅度 (default: 0.65)'
    )
    
    parser.add_argument(
        '--range_threshold',
        type=float,
        default=0.8,
        help='区间判断阈值 (%) - 最高低比率 (default: 0.8)'
    )
    
    parser.add_argument(
        '--min_range_bars',
        type=int,
        default=20,
        help='最低区间长度 (根数, default: 20)'
    )
    
    parser.add_argument(
        '--lookforward',
        type=int,
        default=10,
        help='向前看验证幅度 (default: 10)'
    )
    
    parser.add_argument(
        '--detect_mode',
        type=str,
        default='hybrid',
        choices=['hybrid', 'trend', 'range'],
        help='检测模式 (default: hybrid)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件名'
    )
    
    return parser.parse_args()

def detect_range_zones(df, range_threshold, min_range_bars):
    """
    检测整个区间 (sequential range zones)
    不是滑动窗口，而是找一个接一个的算法
    """
    range_zones = []
    i = 0
    
    while i < len(df):
        # 开始扫描是否是区间
        high = df.iloc[i]['high']
        low = df.iloc[i]['low']
        start_idx = i
        j = i + 1
        
        # 推底区间的上下界
        while j < len(df):
            current_high = df.iloc[j]['high']
            current_low = df.iloc[j]['low']
            
            # 不断更新区间边界
            high = max(high, current_high)
            low = min(low, current_low)
            
            # 检查区间是否超了阻值
            ratio = ((high - low) / low) * 100
            
            if ratio >= range_threshold:
                # 区间破袭，结束
                break
            
            j += 1
        
        # 检查区间是否有效
        range_length = j - start_idx
        
        if range_length >= min_range_bars:
            ratio = ((high - low) / low) * 100
            range_zones.append({
                'start_idx': start_idx,
                'end_idx': j - 1,
                'length': range_length,
                'high': high,
                'low': low,
                'range': high - low,
                'range_percent': ratio,
                'midpoint': (high + low) / 2
            })
        
        # 下一个区间从削掉的位置开始
        i = j if j > i + 1 else i + 1
    
    return range_zones

def detect_trend_reversals(df, lookback, amplitude_threshold, lookforward):
    """
    检测趋势段的反转点
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
    range_threshold = args.range_threshold
    min_range_bars = args.min_range_bars
    detect_mode = args.detect_mode
    lookforward = args.lookforward
    
    print("="*70)
    print("PHASE1: Range Zone Detection with Improved Visualization")
    print("="*70)
    
    print(f"\n[参数设置]")
    print(f"  --lookback {lookback}")
    print(f"  --amplitude_threshold {amplitude_threshold}%")
    print(f"  --range_threshold {range_threshold}%")
    print(f"  --min_range_bars {min_range_bars}")
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
    
    # 初始化标记
    labeled_df = df.copy()
    labeled_df['Zone_Type'] = 'None'  # Trend or Range
    labeled_df['Label_Type'] = 'None'  # Local_High/Low or Range_Start/End
    labeled_df['Label_Value'] = 0.0
    labeled_df['Label_Strength'] = 0.0
    
    print("\n[2/4] Detecting range zones...")
    
    # 检测区间
    range_zones = detect_range_zones(df, range_threshold, min_range_bars)
    print(f"Detected {len(range_zones)} range zones")
    
    # 一开始，所有都是 Trend
    labeled_df['Zone_Type'] = 'Trend'
    
    # 控制区间
    for zone in range_zones:
        start = int(zone['start_idx'])
        end = int(zone['end_idx']) + 1
        labeled_df.loc[start:end, 'Zone_Type'] = 'Range'
    
    trend_count = (labeled_df['Zone_Type'] == 'Trend').sum()
    range_count = (labeled_df['Zone_Type'] == 'Range').sum()
    print(f"Trend bars: {trend_count}, Range bars: {range_count}")
    print(f"Trend ratio: {trend_count/len(df)*100:.1f}%, Range ratio: {range_count/len(df)*100:.1f}%")
    
    trend_reversals = 0
    range_starts = 0
    range_ends = 0
    
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
    
    if detect_mode in ['range', 'hybrid']:
        print("  Marking range zone boundaries...")
        
        for zone in range_zones:
            start = int(zone['start_idx'])
            end = int(zone['end_idx'])
            
            # 区间开始 (buy signal)
            labeled_df.at[start, 'Label_Type'] = 'Range_Start'
            labeled_df.at[start, 'Label_Value'] = zone['low']
            labeled_df.at[start, 'Label_Strength'] = zone['range_percent']
            range_starts += 1
            
            # 区间结束 (sell signal)
            labeled_df.at[end, 'Label_Type'] = 'Range_End'
            labeled_df.at[end, 'Label_Value'] = zone['high']
            labeled_df.at[end, 'Label_Strength'] = zone['range_percent']
            range_ends += 1
        
        print(f"  Marked {range_starts} range starts + {range_ends} range ends")
    
    # 保存CSV
    if args.output:
        output_file = args.output
    else:
        output_file = f'phase1_range_lb{lookback}_amp{amplitude_threshold}_rt{range_threshold}.csv'
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Zone_Type', 'Label_Type', 'Label_Value', 'Label_Strength']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    labeled_df[existing_cols].to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("PHASE1 Range Detection Results")
    print(f"{'='*70}")
    print(f"\nZone Statistics:")
    print(f"  Trend bars: {trend_count} ({trend_count/len(df)*100:.1f}%)")
    print(f"  Range bars: {range_count} ({range_count/len(df)*100:.1f}%)")
    
    if detect_mode in ['trend', 'hybrid']:
        print(f"\nTrend Reversals: {trend_reversals}")
    if detect_mode in ['range', 'hybrid']:
        print(f"\nRange Zones: {len(range_zones)}")
        print(f"  Range starts: {range_starts}")
        print(f"  Range ends: {range_ends}")
        if range_zones:
            print(f"  Avg range length: {np.mean([z['length'] for z in range_zones]):.0f} bars")
            print(f"  Avg range width: {np.mean([z['range_percent'] for z in range_zones]):.2f}%")
    
    print(f"\n{'='*70}")
    print("[4/4] Generating visualization...")
    print(f"{'='*70}")
    
    # 可视化
    rcParams['figure.figsize'] = (16, 8)
    start_idx = max(0, len(labeled_df) - 800)
    plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # K线圖
    ax1.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1.5, label='Close Price', zorder=1)
    
    # 区间背景整店
    range_mask = plot_df['Zone_Type'] == 'Range'
    ax1.fill_between(range(len(plot_df)), plot_df['close'].min()-100, plot_df['close'].max()+100,
                      where=range_mask, alpha=0.15, color='orange', label='Range Zone', zorder=0)
    
    # 趋势反转点
    if detect_mode in ['trend', 'hybrid']:
        highs = plot_df[(plot_df['Label_Type'] == 'Local_High')]
        lows = plot_df[(plot_df['Label_Type'] == 'Local_Low')]
        ax1.scatter(highs.index, highs['close'], color='red', marker='v', s=150, label=f'Sell ({len(highs)})', zorder=5, edgecolors='darkred', linewidth=1.5)
        ax1.scatter(lows.index, lows['close'], color='green', marker='^', s=150, label=f'Buy ({len(lows)})', zorder=5, edgecolors='darkgreen', linewidth=1.5)
    
    # 区间像麵（大昭段竖线）
    if detect_mode in ['range', 'hybrid']:
        range_starts = plot_df[(plot_df['Label_Type'] == 'Range_Start')]
        range_ends = plot_df[(plot_df['Label_Type'] == 'Range_End')]
        
        # 区间开始 - 绿色符号
        for idx in range_starts.index:
            ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(idx, plot_df['close'].max(), '\u25b2', ha='center', fontsize=12, color='green')
        
        # 区间结束 - 红色符号
        for idx in range_ends.index:
            ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax1.text(idx, plot_df['close'].max(), '\u25bc', ha='center', fontsize=12, color='red')
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PHASE1 Range Zone Detection - range_threshold={range_threshold}%, min_bars={min_range_bars}\nTrend: {trend_count} | Range: {range_count}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 区间类型下方
    zone_colors = plot_df['Zone_Type'].map({'Trend': 0, 'Range': 1})
    ax2.fill_between(range(len(plot_df)), 0, zone_colors, alpha=0.5, color='steelblue')
    ax2.set_ylabel('Zone Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('K-line Index', fontsize=12, fontweight='bold')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Trend', 'Range'])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    png_file = output_file.replace('.csv', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {png_file}")
    plt.show()
    
    print(f"\n{'='*70}")
    print("PHASE1 Range Detection Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
