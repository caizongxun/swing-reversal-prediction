#!/usr/bin/env python3
"""
PHASE1 - Bollinger Bands Range Detection

Using Bollinger Bands to detect sideways ranges accurately

Usage in Colab:

!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_bollinger_bands.py -O phase1_bb.py

# 例1: 混合模式 (趋势反转 + BB横盘)
!python3 phase1_bb.py --lookback 8 --amplitude_threshold 0.65 --bb_period 20 --bb_std 2.0 --detect_mode hybrid

# 例2: 仅趋势
!python3 phase1_bb.py --lookback 8 --amplitude_threshold 0.65 --detect_mode trend

# 例3: 仅横盘 (BB检测)
!python3 phase1_bb.py --bb_period 20 --bb_std 2.0 --detect_mode sideways

Bollinger Bands原理:
  中线 = SMA(close, period)
  上轨 = 中线 + std_dev * std
  下轨 = 中线 - std_dev * std
  
  横盘判定: price在上下轨之间振荡,不突破
  趋势判定: price穿过上/下轨,突破

bb_period: 计算周期 (通常20)
bb_std: 标准差倍数 (通常2.0)
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
        description='PHASE1 - Bollinger Bands Range Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 phase1_bb.py --lookback 8 --amplitude_threshold 0.65 --bb_period 20 --bb_std 2.0 --detect_mode hybrid
  python3 phase1_bb.py --bb_period 20 --bb_std 2.0 --detect_mode sideways
  python3 phase1_bb.py --lookback 8 --amplitude_threshold 0.65 --detect_mode trend
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
        '--bb_period',
        type=int,
        default=20,
        help='布林通道周期 (default: 20)'
    )
    
    parser.add_argument(
        '--bb_std',
        type=float,
        default=2.0,
        help='布林通道标准差倍数 (default: 2.0)'
    )
    
    parser.add_argument(
        '--bb_squeeze_threshold',
        type=float,
        default=0.1,
        help='布林带压缩判定阈值 - 带宽/中线 (default: 0.1 = 10%)'
    )
    
    parser.add_argument(
        '--min_sideways_bars',
        type=int,
        default=15,
        help='最小横盘长度 (根数, default: 15)'
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

def calculate_bollinger_bands(df, period=20, std_multiplier=2.0):
    """
    计算布林通道
    """
    df = df.copy()
    
    # 计算中线 (简单移动平均)
    df['bb_middle'] = df['close'].rolling(window=period).mean()
    
    # 计算标准差
    df['bb_std'] = df['close'].rolling(window=period).std()
    
    # 计算上下轨
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_multiplier)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_multiplier)
    
    # 计算带宽 (上轨 - 下轨)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # 计算带宽百分比 (带宽 / 中线) - 用于检测压缩
    df['bb_width_pct'] = (df['bb_width'] / df['bb_middle']) * 100
    
    # 判断价格位置
    # 0 = 下方, 1 = 中间, 2 = 上方
    df['bb_position'] = 1  # 默认中间
    df.loc[df['close'] < df['bb_lower'], 'bb_position'] = 0  # 下轨下方
    df.loc[df['close'] > df['bb_upper'], 'bb_position'] = 2  # 上轨上方
    
    return df

def detect_sideways_by_bollinger(df, squeeze_threshold, min_sideways_bars):
    """
    用布林通道检测横盘区间
    
    横盘特征:
    1. 价格在上下轨之间振荡 (bb_position == 1)
    2. 布林带宽缩小 (带宽压缩)
    3. 持续至少 min_sideways_bars 根K线
    """
    sideways_zones = []
    i = 0
    
    while i < len(df):
        if pd.isna(df.iloc[i]['bb_middle']):
            i += 1
            continue
        
        # 检查是否在带内
        if df.iloc[i]['bb_position'] == 1:  # 价格在中间
            start_idx = i
            j = i
            in_band_count = 0
            
            # 向前扩展，找到整个横盘区间
            while j < len(df):
                if pd.isna(df.iloc[j]['bb_middle']):
                    break
                
                # 价格仍在带内
                if df.iloc[j]['bb_position'] == 1:
                    in_band_count += 1
                    j += 1
                else:
                    # 价格突破带
                    break
            
            sideways_length = j - start_idx
            
            # 检查是否满足最小长度
            if sideways_length >= min_sideways_bars:
                # 计算这个区间的统计信息
                zone_df = df.iloc[start_idx:j]
                high = zone_df['high'].max()
                low = zone_df['low'].min()
                range_pct = ((high - low) / low) * 100
                avg_bb_width = zone_df['bb_width_pct'].mean()
                
                sideways_zones.append({
                    'start_idx': start_idx,
                    'end_idx': j - 1,
                    'length': sideways_length,
                    'high': high,
                    'low': low,
                    'range_pct': range_pct,
                    'bb_width_avg': avg_bb_width,
                    'bb_upper': zone_df['bb_upper'].iloc[0],
                    'bb_lower': zone_df['bb_lower'].iloc[0],
                    'bb_middle': zone_df['bb_middle'].iloc[0]
                })
            
            i = j if j > i + 1 else i + 1
        else:
            i += 1
    
    return sideways_zones

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
    bb_period = args.bb_period
    bb_std = args.bb_std
    squeeze_threshold = args.bb_squeeze_threshold
    min_sideways_bars = args.min_sideways_bars
    detect_mode = args.detect_mode
    lookforward = args.lookforward
    
    print("="*70)
    print("PHASE1: Bollinger Bands Range Detection")
    print("="*70)
    
    print(f"\n[参数设置]")
    print(f"  Bollinger Bands:")
    print(f"    --bb_period {bb_period}")
    print(f"    --bb_std {bb_std}")
    print(f"  趋势检测:")
    print(f"    --lookback {lookback}")
    print(f"    --amplitude_threshold {amplitude_threshold}%")
    print(f"  模式: --detect_mode {detect_mode}")
    
    print("\n[1/5] Loading data...")
    
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
    
    print(f"\n[2/5] Calculating Bollinger Bands...")
    df = calculate_bollinger_bands(df, bb_period, bb_std)
    print(f"Bollinger Bands calculated (period={bb_period}, std={bb_std})")
    
    # 初始化标记列
    labeled_df = df.copy()
    labeled_df['Market_Mode'] = 'Unknown'
    labeled_df['Label_Type'] = 'None'
    labeled_df['Label_Value'] = 0.0
    labeled_df['Label_Strength'] = 0.0
    labeled_df['BB_Upper'] = labeled_df['bb_upper']
    labeled_df['BB_Middle'] = labeled_df['bb_middle']
    labeled_df['BB_Lower'] = labeled_df['bb_lower']
    
    print(f"\n[3/5] Detecting sideways zones (Bollinger Bands)...")
    sideways_zones = detect_sideways_by_bollinger(df, squeeze_threshold, min_sideways_bars)
    print(f"Detected {len(sideways_zones)} sideways zones")
    
    # 默认都是Trend
    labeled_df['Market_Mode'] = 'Trend'
    
    # 标记横盘区间
    for zone in sideways_zones:
        start = int(zone['start_idx'])
        end = int(zone['end_idx']) + 1
        labeled_df.loc[start:end, 'Market_Mode'] = 'Sideways'
    
    trend_count = (labeled_df['Market_Mode'] == 'Trend').sum()
    sideways_count = (labeled_df['Market_Mode'] == 'Sideways').sum()
    print(f"Trend bars: {trend_count} ({trend_count/len(df)*100:.1f}%)")
    print(f"Sideways bars: {sideways_count} ({sideways_count/len(df)*100:.1f}%)")
    
    trend_reversals = 0
    sideways_marks = 0
    
    print(f"\n[4/5] Labeling...")
    
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
        print("  Marking sideways zones...")
        
        for i, zone in enumerate(sideways_zones):
            start = int(zone['start_idx'])
            end = int(zone['end_idx'])
            
            # 标记区间开始 (绿色, 买点)
            labeled_df.at[start, 'Label_Type'] = 'Sideways_Start'
            labeled_df.at[start, 'Label_Value'] = zone['low']
            labeled_df.at[start, 'Label_Strength'] = zone['range_pct']
            
            # 标记区间结束 (红色, 卖点)
            labeled_df.at[end, 'Label_Type'] = 'Sideways_End'
            labeled_df.at[end, 'Label_Value'] = zone['high']
            labeled_df.at[end, 'Label_Strength'] = zone['range_pct']
            
            sideways_marks += 2
        
        print(f"  Marked {len(sideways_zones)} sideways zones")
    
    # 保存CSV
    if args.output:
        output_file = args.output
    else:
        output_file = f'phase1_bb_period{bb_period}_std{bb_std}.csv'
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'bb_width_pct',
            'Market_Mode', 'Label_Type', 'Label_Value', 'Label_Strength']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    labeled_df[existing_cols].to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    print(f"\n{'='*70}")
    print("PHASE1 Bollinger Bands Detection Results")
    print(f"{'='*70}")
    print(f"\nZone Statistics:")
    print(f"  Trend bars: {trend_count} ({trend_count/len(df)*100:.1f}%)")
    print(f"  Sideways bars: {sideways_count} ({sideways_count/len(df)*100:.1f}%)")
    
    if detect_mode in ['trend', 'hybrid']:
        print(f"\nTrend Reversals: {trend_reversals}")
    if detect_mode in ['sideways', 'hybrid']:
        print(f"\nSideways Zones: {len(sideways_zones)}")
        if sideways_zones:
            print(f"  Avg zone length: {np.mean([z['length'] for z in sideways_zones]):.0f} bars")
            print(f"  Avg zone range: {np.mean([z['range_pct'] for z in sideways_zones]):.2f}%")
            print(f"  Avg BB width: {np.mean([z['bb_width_avg'] for z in sideways_zones]):.2f}%")
    
    print(f"\n{'='*70}")
    print("[5/5] Generating visualization...")
    print(f"{'='*70}")
    
    # 可视化
    rcParams['figure.figsize'] = (16, 9)
    start_idx = max(0, len(labeled_df) - 800)
    plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # K线图
    ax1.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1.5, label='Close Price', zorder=1)
    
    # 布林通道
    ax1.fill_between(range(len(plot_df)), plot_df['BB_Lower'], plot_df['BB_Upper'],
                      alpha=0.15, color='blue', label='Bollinger Bands', zorder=0)
    ax1.plot(range(len(plot_df)), plot_df['BB_Upper'], color='blue', linewidth=1, linestyle='--', alpha=0.5)
    ax1.plot(range(len(plot_df)), plot_df['BB_Lower'], color='blue', linewidth=1, linestyle='--', alpha=0.5)
    ax1.plot(range(len(plot_df)), plot_df['BB_Middle'], color='blue', linewidth=1, linestyle='-', alpha=0.3)
    
    # 横盘背景
    sideways_mask = plot_df['Market_Mode'] == 'Sideways'
    ax1.fill_between(range(len(plot_df)), plot_df['close'].min()-100, plot_df['close'].max()+100,
                      where=sideways_mask, alpha=0.1, color='orange', label='Sideways Zone', zorder=0)
    
    # 趋势反转点
    if detect_mode in ['trend', 'hybrid']:
        highs = plot_df[(plot_df['Label_Type'] == 'Local_High')]
        lows = plot_df[(plot_df['Label_Type'] == 'Local_Low')]
        ax1.scatter(highs.index, highs['close'], color='red', marker='v', s=150, label=f'Sell ({len(highs)})', zorder=5, edgecolors='darkred', linewidth=1.5)
        ax1.scatter(lows.index, lows['close'], color='green', marker='^', s=150, label=f'Buy ({len(lows)})', zorder=5, edgecolors='darkgreen', linewidth=1.5)
    
    # 横盘标记
    if detect_mode in ['sideways', 'hybrid']:
        starts = plot_df[(plot_df['Label_Type'] == 'Sideways_Start')]
        ends = plot_df[(plot_df['Label_Type'] == 'Sideways_End')]
        
        for idx in starts.index:
            ax1.axvline(x=idx, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
            ax1.text(idx, plot_df['close'].max(), '▲', ha='center', fontsize=14, color='green', fontweight='bold')
        
        for idx in ends.index:
            ax1.axvline(x=idx, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
            ax1.text(idx, plot_df['close'].max(), '▼', ha='center', fontsize=14, color='red', fontweight='bold')
    
    ax1.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'PHASE1 Bollinger Bands Detection (BB_period={bb_period}, std={bb_std})\nTrend: {trend_count} | Sideways: {sideways_count}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # 市场模式
    mode_colors = plot_df['Market_Mode'].map({'Trend': 0, 'Sideways': 1})
    ax2.fill_between(range(len(plot_df)), 0, mode_colors, alpha=0.5, color='steelblue')
    ax2.set_ylabel('Mode', fontsize=12, fontweight='bold')
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
    print("PHASE1 Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
