#!/usr/bin/env python3
"""
PHASE1 Direction Change Detection - With Amplitude Threshold Filter

Usage in Colab:

!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_amplitude_filter.py -O phase1.py

# Example 1: lookback=10, confirm_local=True, amplitude_threshold=0.5% (滤掉假信号)
!python3 phase1.py --lookback 10 --confirm_local True --amplitude_threshold 0.5

# Example 2: lookback=10, amplitude_threshold=1% (更严格)
!python3 phase1.py --lookback 10 --amplitude_threshold 1.0

# Example 3: lookback=5, amplitude_threshold=0.3% (较宽松)
!python3 phase1.py --lookback 5 --amplitude_threshold 0.3

Amplitude_threshold说明:
  - 反转后需要达到最少X%的涨跌幅才被认为是有效反转
  - 越大=噪音少但会遗漏小反转
  - 越小=捕捉更多但可能有假信号
  - 推荐: 0.5% - 1.0%
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
        description='PHASE1 Direction Change Detection with Amplitude Filter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 phase1.py --lookback 10 --confirm_local True --amplitude_threshold 0.5
  python3 phase1.py --lookback 10 --amplitude_threshold 1.0
  python3 phase1.py --lookback 5 --amplitude_threshold 0.3
        """
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=10,
        help='向后看N根K线来确定方向 (default: 10, range: 1-10)'
    )
    
    parser.add_argument(
        '--confirm_local',
        type=str,
        default='True',
        choices=['True', 'False', 'true', 'false'],
        help='是否验证局部高低点 (default: True)'
    )
    
    parser.add_argument(
        '--amplitude_threshold',
        type=float,
        default=0.5,
        help='反转后需要的最低涨跌幅(%) (default: 0.5, range: 0.1-2.0)'
    )
    
    parser.add_argument(
        '--lookforward',
        type=int,
        default=10,
        help='向前看N根K线来验证涨跌幅 (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件名 (default: auto-generate)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 转换参数
    lookback = args.lookback
    confirm_local = args.confirm_local.lower() == 'true'
    amplitude_threshold = args.amplitude_threshold
    lookforward = args.lookforward
    
    # 验证参数范围
    if lookback < 1 or lookback > 10:
        print(f"ERROR: lookback must be between 1 and 10, got {lookback}")
        sys.exit(1)
    
    if amplitude_threshold < 0.1 or amplitude_threshold > 2.0:
        print(f"WARNING: amplitude_threshold should be 0.1-2.0, got {amplitude_threshold}. 按传继。")
    
    print("="*70)
    print("PHASE1: Direction Change with Amplitude Filter")
    print("="*70)
    
    print(f"\n[参数设置]")
    print(f"  --lookback {lookback}")
    print(f"  --confirm_local {confirm_local}")
    print(f"  --amplitude_threshold {amplitude_threshold}%")
    print(f"  --lookforward {lookforward}")
    
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
    
    print(f"数据大小: {df.shape}")
    print(f"时间范围: {str(df['timestamp'].iloc[0])[:19]} 至 {str(df['timestamp'].iloc[-1])[:19]}")
    
    print(f"\n[2/4] 检测反转点...")
    
    reversals = []
    amplitude_data = []  # 记录涨跌幅数据供参考
    
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
        """
        检查反转后是否有足够的涨跌幅
        direction_type: 'up' 或 'down'
        threshold: 涨跌幅阈值 (%)
        look_ahead: 向前看计算个根数
        """
        reversal_price = df.iloc[idx]['close']
        max_idx = min(idx + look_ahead, len(df) - 1)
        
        if direction_type == 'up':  # Low反转, 之后需要涨
            max_price = df.iloc[idx:max_idx+1]['high'].max()
            amplitude = ((max_price - reversal_price) / reversal_price) * 100
        else:  # High反转, 之后需要跌
            min_price = df.iloc[idx:max_idx+1]['low'].min()
            amplitude = ((reversal_price - min_price) / reversal_price) * 100
        
        return amplitude, amplitude >= threshold
    
    for i in range(lookback + 1, len(df) - lookforward):
        prev_direction = calculate_direction(i-1, lookback)
        current_direction = calculate_direction(i, 1)
        
        # 检测高点反转(从上升变下降)
        if prev_direction > 0 and current_direction < 0:
            if confirm_local and not is_local_high(i-1):
                continue
            
            # 检查跌幅是否质量高
            amplitude, is_valid = check_amplitude(i-1, 'down', amplitude_threshold, lookforward)
            
            if is_valid:
                reversals.append({
                    'index': i-1,
                    'timestamp': df.iloc[i-1]['timestamp'],
                    'close': df.iloc[i-1]['close'],
                    'type': 'Local_High',
                    'amplitude': amplitude
                })
                amplitude_data.append({
                    'type': 'High',
                    'amplitude': amplitude,
                    'valid': True
                })
            else:
                amplitude_data.append({
                    'type': 'High',
                    'amplitude': amplitude,
                    'valid': False
                })
        
        # 检测低点反转(从下跌变上升)
        elif prev_direction < 0 and current_direction > 0:
            if confirm_local and not is_local_low(i-1):
                continue
            
            # 检查涨幅是否质量高
            amplitude, is_valid = check_amplitude(i-1, 'up', amplitude_threshold, lookforward)
            
            if is_valid:
                reversals.append({
                    'index': i-1,
                    'timestamp': df.iloc[i-1]['timestamp'],
                    'close': df.iloc[i-1]['close'],
                    'type': 'Local_Low',
                    'amplitude': amplitude
                })
                amplitude_data.append({
                    'type': 'Low',
                    'amplitude': amplitude,
                    'valid': True
                })
            else:
                amplitude_data.append({
                    'type': 'Low',
                    'amplitude': amplitude,
                    'valid': False
                })
    
    print(f"检测到 {len(reversals)} 个有效反转点")
    
    # 统计幷民率
    amplitude_df = pd.DataFrame(amplitude_data) if amplitude_data else pd.DataFrame()
    if len(amplitude_df) > 0:
        invalid_count = len(amplitude_df[amplitude_df['valid'] == False])
        print(f"筛选掉 {invalid_count} 个假信号 (涨跌幅 < {amplitude_threshold}%)")
    
    print("\n[3/4] 标记数据集...")
    
    labeled_df = df.copy()
    labeled_df['Reversal_Label'] = 0
    labeled_df['Reversal_Type'] = 'None'
    labeled_df['Reversal_Price'] = 0.0
    labeled_df['Reversal_Amplitude'] = 0.0
    
    for reversal in reversals:
        i = int(reversal['index'])
        if i < len(labeled_df):
            labeled_df.at[i, 'Reversal_Label'] = 1
            labeled_df.at[i, 'Reversal_Type'] = reversal['type']
            labeled_df.at[i, 'Reversal_Price'] = reversal['close']
            labeled_df.at[i, 'Reversal_Amplitude'] = reversal['amplitude']
    
    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        output_file = f'phase1_lb{lookback}_amp{amplitude_threshold}.csv'
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Reversal_Label', 'Reversal_Type', 'Reversal_Price', 'Reversal_Amplitude']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    labeled_df[existing_cols].to_csv(output_file, index=False)
    
    print(f"已保存至: {output_file}")
    
    reversals_df = pd.DataFrame(reversals) if reversals else pd.DataFrame()
    total = len(reversals)
    local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low']) if total > 0 else 0
    local_highs = len(reversals_df[reversals_df['type'] == 'Local_High']) if total > 0 else 0
    
    print(f"\n{'='*70}")
    print("PHASE1 Detection Results (with Amplitude Filter)")
    print(f"{'='*70}")
    
    print(f"\n有效反转点数: {total}")
    print(f"  - 局部低点 (买入点): {local_lows}")
    print(f"  - 局部高点 (卖出点): {local_highs}")
    print(f"\n反转点占比: {total / len(df) * 100:.2f}%")
    if total > 0:
        print(f"平均间距: {len(df) // (total + 1):.0f} 根K线")
        if 'amplitude' in reversals_df.columns:
            print(f"平均涨跌幅: {reversals_df['amplitude'].mean():.2f}%")
    
    if total > 0:
        print(f"\n前10个有效反转点:")
        print(reversals_df[['index', 'timestamp', 'type', 'close', 'amplitude']].head(10).to_string())
    
    print(f"\n{'='*70}")
    print("[4/4] 生成可视化图表...")
    print(f"{'='*70}")
    
    if total > 0:
        rcParams['figure.figsize'] = (16, 7)
        
        start_idx = max(0, len(labeled_df) - 800)
        plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(16, 7))
        
        ax.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1.5, label='Close Price', zorder=1)
        
        lows = plot_df[plot_df['Reversal_Type'] == 'Local_Low']
        ax.scatter(lows.index, lows['close'], color='green', marker='^', s=150, label=f'Local Low ({len(lows)})', zorder=5, edgecolors='darkgreen', linewidth=1.5)
        
        highs = plot_df[plot_df['Reversal_Type'] == 'Local_High']
        ax.scatter(highs.index, highs['close'], color='red', marker='v', s=150, label=f'Local High ({len(highs)})', zorder=5, edgecolors='darkred', linewidth=1.5)
        
        ax.set_xlabel('K线索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('价格 (USDT)', fontsize=12, fontweight='bold')
        ax.set_title(f'PHASE1 反转点检测 (With Amplitude Filter) - lookback={lookback}, amplitude>={amplitude_threshold}%\nTotal: {len(plot_df[plot_df["Reversal_Label"]==1])} valid reversals in recent 800 bars', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        png_file = output_file.replace('.csv', '.png')
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存: {png_file}")
        plt.show()
        
        print(f"\n最近800根K线中检测到 {len(plot_df[plot_df['Reversal_Label']==1])} 个有效反转点")
    else:
        print("\n未检测到有效反转点 (阈值可能过高)")
    
    print(f"\n{'='*70}")
    print(f"PHASE1 Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
