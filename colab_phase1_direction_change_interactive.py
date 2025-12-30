#!/usr/bin/env python3
"""
PHASE1 Direction Change Detection - Interactive Colab Version

Usage in Colab cell:
!python colab_phase1_direction_change_interactive.py --lookback 5 --confirm_local True

Parameters:
  --lookback INT          : 向后看N根K线来确定方向 (default: 3, range: 1-10)
  --confirm_local BOOL    : 是否验证局部高低点 (default: True)
  --output FILE           : 输出文件名 (default: labeled_klines_phase1_direction_change.csv)

Examples:
  !python colab_phase1_direction_change_interactive.py --lookback 5 --confirm_local True
  !python colab_phase1_direction_change_interactive.py --lookback 3 --confirm_local False
  !python colab_phase1_direction_change_interactive.py --lookback 7
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='PHASE1 Direction Change Detection - Interactive',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=5,
        help='向后看N根K线来确定方向 (default: 5, range: 1-10)'
    )
    
    parser.add_argument(
        '--confirm_local',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='是否验证局部高低点 (default: True)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='labeled_klines_phase1_direction_change.csv',
        help='输出文件名 (default: labeled_klines_phase1_direction_change.csv)'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 验证参数
    if args.lookback < 1 or args.lookback > 10:
        print(f"ERROR: lookback must be between 1 and 10, got {args.lookback}")
        sys.exit(1)
    
    print("="*70)
    print("PHASE1: Direction Change Detection (Interactive)")
    print("="*70)
    
    print(f"\n参数设置:")
    print(f"  lookback: {args.lookback}")
    print(f"  confirm_local: {args.confirm_local}")
    print(f"  output: {args.output}")
    
    print("\n加载数据...")
    
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
        if os.path.exists('labeled_klines_phase1.csv'):
            df = pd.read_csv('labeled_klines_phase1.csv')
            print("已从labeled_klines_phase1.csv加载")
        else:
            print("ERROR: 未找到数据文件")
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
    
    print(f"\n数据大小: {df.shape}")
    print(f"时间范围: {str(df['timestamp'].iloc[0])[:19]} 至 {str(df['timestamp'].iloc[-1])[:19]}")
    
    lookback = args.lookback
    confirm_local = args.confirm_local
    
    print(f"\n检测反转点 (lookback={lookback}, confirm_local={confirm_local})...")
    
    reversals = []
    
    def calculate_direction(idx, n):
        if idx < n:
            return 0
        start_close = df.iloc[idx-n]['close']
        end_close = df.iloc[idx]['close']
        return end_close - start_close
    
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
    
    print("\n[1/2] 检测方向改变...")
    
    for i in range(lookback + 1, len(df) - 1):
        prev_direction = calculate_direction(i-1, lookback)
        current_direction = calculate_direction(i, 1)
        
        # 高点反转: 从上升变下降
        if prev_direction > 0 and current_direction < 0:
            if confirm_local and not is_local_high(i-1):
                continue
            reversals.append({
                'index': i-1,
                'timestamp': df.iloc[i-1]['timestamp'],
                'close': df.iloc[i-1]['close'],
                'type': 'Local_High'
            })
        
        # 低点反转: 从下降变上升
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
    
    print("\n[2/2] 标记数据集...")
    
    df['Reversal_Label'] = 0
    df['Reversal_Type'] = 'None'
    df['Reversal_Price'] = 0.0
    
    for reversal in reversals:
        i = int(reversal['index'])
        if i < len(df):
            df.at[i, 'Reversal_Label'] = 1
            df.at[i, 'Reversal_Type'] = reversal['type']
            df.at[i, 'Reversal_Price'] = reversal['close']
    
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
    existing_cols = [c for c in cols if c in df.columns]
    df[existing_cols].to_csv(args.output, index=False)
    
    print(f"\n已保存至: {args.output}")
    
    # 统计
    reversals_df = pd.DataFrame(reversals) if reversals else pd.DataFrame()
    total = len(reversals)
    local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low']) if len(reversals_df) > 0 else 0
    local_highs = len(reversals_df[reversals_df['type'] == 'Local_High']) if len(reversals_df) > 0 else 0
    
    print(f"\n{'='*70}")
    print("PHASE1 检测结果")
    print(f"{'='*70}")
    
    print(f"\n总反转点数: {total}")
    print(f"  - 局部低点 (买入点): {local_lows}")
    print(f"  - 局部高点 (卖出点): {local_highs}")
    print(f"\n反转点占比: {total / len(df) * 100:.2f}%")
    if total > 0:
        print(f"平均间距: {len(df) // (total + 1):.0f} 根K线")
    
    if len(reversals_df) > 0:
        print(f"\n前15个反转点:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].head(15).to_string())
        
        print(f"\n最后15个反转点:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].tail(15).to_string())
    else:
        print(f"\n未检测到反转点")
    
    print(f"\n{'='*70}")
    print("PHASE1 完成!")
    print(f"{'='*70}")
    
    return df, reversals_df

if __name__ == '__main__':
    main()
