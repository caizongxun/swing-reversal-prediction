#!/usr/bin/env python3
"""
PHASE1: Direction Change Detection Method

Core Logic (方向改变法):
1. 计算过去N根K线的平均方向(上升/下降)
2. 检测当前K线是否改变方向
3. 从上升转下降 -> 标记为High反转点
4. 从下跌转上升 -> 标记为Low反转点

这个方法能准确捕捉实际的买卖转折点。
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DirectionChangeDetector:
    """
    基于方向改变的反转点检测器
    """
    
    def __init__(self, df, lookback=3, lookforward=3, confirm_local=True):
        """
        Args:
            df: DataFrame with OHLCV data
            lookback: 检查过去N根K线的方向
            lookforward: 验证未来N根K线的方向
            confirm_local: 是否验证局部相对高低点
        """
        self.df = df.copy()
        self.lookback = lookback
        self.lookforward = lookforward
        self.confirm_local = confirm_local
        self.reversals = []
        
    def calculate_direction(self, idx, n):
        """
        计算从idx-n到idx的方向
        返回值: 正数=上升, 负数=下降, 0=平
        """
        if idx < n:
            return 0
        
        start_close = self.df.iloc[idx-n]['close']
        end_close = self.df.iloc[idx]['close']
        
        return end_close - start_close
    
    def is_local_high(self, idx):
        """
        检查是否为局部相对高点
        close[i] >= close[i-1] AND close[i] >= close[i+1]
        """
        if idx <= 0 or idx >= len(self.df) - 1:
            return False
        
        current = self.df.iloc[idx]['close']
        prev = self.df.iloc[idx-1]['close']
        next_val = self.df.iloc[idx+1]['close']
        
        return current >= prev and current >= next_val
    
    def is_local_low(self, idx):
        """
        检查是否为局部相对低点
        close[i] <= close[i-1] AND close[i] <= close[i+1]
        """
        if idx <= 0 or idx >= len(self.df) - 1:
            return False
        
        current = self.df.iloc[idx]['close']
        prev = self.df.iloc[idx-1]['close']
        next_val = self.df.iloc[idx+1]['close']
        
        return current <= prev and current <= next_val
    
    def detect_reversals(self):
        """
        检测所有反转点
        """
        print(f"[1/3] 计算方向变化 (lookback={self.lookback})...")
        
        for i in range(self.lookback + 1, len(self.df) - self.lookforward):
            # 计算过去方向
            prev_direction = self.calculate_direction(i-1, self.lookback)
            
            # 计算当前方向
            current_direction = self.calculate_direction(i, 1)
            
            # 检测高点反转(从上升变下降)
            if prev_direction > 0 and current_direction < 0:
                # 如果启用本地验证
                if self.confirm_local and not self.is_local_high(i-1):
                    continue
                
                self.reversals.append({
                    'index': i-1,
                    'timestamp': self.df.iloc[i-1]['timestamp'],
                    'close': self.df.iloc[i-1]['close'],
                    'type': 'Local_High',
                    'prev_direction': prev_direction,
                    'current_direction': current_direction
                })
            
            # 检测低点反转(从下跌变上升)
            elif prev_direction < 0 and current_direction > 0:
                # 如果启用本地验证
                if self.confirm_local and not self.is_local_low(i-1):
                    continue
                
                self.reversals.append({
                    'index': i-1,
                    'timestamp': self.df.iloc[i-1]['timestamp'],
                    'close': self.df.iloc[i-1]['close'],
                    'type': 'Local_Low',
                    'prev_direction': prev_direction,
                    'current_direction': current_direction
                })
        
        print(f"   发现 {len(self.reversals)} 个反转点")
        
        return pd.DataFrame(self.reversals).sort_values('index').reset_index(drop=True) if self.reversals else pd.DataFrame()
    
    def label_dataset(self):
        """
        为整个数据集标记反转点
        """
        self.df['Reversal_Label'] = 0
        self.df['Reversal_Type'] = 'None'
        self.df['Reversal_Price'] = 0.0
        
        for idx, row in self.reversals.iterrows():
            i = int(row['index'])
            if i < len(self.df):
                self.df.at[i, 'Reversal_Label'] = 1
                self.df.at[i, 'Reversal_Type'] = row['type']
                self.df.at[i, 'Reversal_Price'] = row['close']
        
        return self.df
    
    def get_summary(self):
        """
        获取统计摘要
        """
        local_lows = pd.DataFrame(self.reversals)[pd.DataFrame(self.reversals)['type'] == 'Local_Low']
        local_highs = pd.DataFrame(self.reversals)[pd.DataFrame(self.reversals)['type'] == 'Local_High']
        
        return {
            'total': len(self.reversals),
            'local_lows': len(local_lows),
            'local_highs': len(local_highs),
            'ratio': len(self.reversals) / len(self.df) * 100
        }


def main():
    print("="*70)
    print("PHASE1: 方向改变法 - 反转点检测")
    print("="*70)
    
    csv_file = 'labeled_klines_phase1.csv'
    
    print(f"\n加载数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 数据类型转换
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    
    print(f"数据大小: {df.shape}")
    print(f"时间范围: {df['timestamp'].iloc[0]} 至 {df['timestamp'].iloc[-1]}")
    
    # 创建检测器
    # 参数: lookback=3, lookforward=3, confirm_local=True
    detector = DirectionChangeDetector(df, lookback=3, lookforward=3, confirm_local=True)
    
    print(f"\n{'='*70}")
    print("执行反转点检测...")
    print(f"{'='*70}")
    print(f"参数: lookback=3, lookforward=3, 启用局部验证")
    
    reversals_df = detector.detect_reversals()
    
    print(f"\n标记数据集...")
    labeled_df = detector.label_dataset()
    
    # 保存结果
    output_file = 'labeled_klines_phase1_direction_change.csv'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    
    labeled_df[existing_cols].to_csv(output_file, index=False)
    print(f"已保存至: {output_file}")
    
    # 统计
    summary = detector.get_summary()
    
    print(f"\n{'='*70}")
    print("PHASE1 检测结果摘要")
    print(f"{'='*70}")
    
    print(f"\n总反转点数: {summary['total']}")
    print(f"  - 局部低点 (买入点): {summary['local_lows']}")
    print(f"  - 局部高点 (卖出点): {summary['local_highs']}")
    print(f"\n反转点占比: {summary['ratio']:.2f}%")
    print(f"平均间距: {len(df) // (summary['total'] + 1):.0f} 根K线")
    
    if len(reversals_df) > 0:
        print(f"\n前15个反转点:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].head(15).to_string())
        
        print(f"\n最后15个反转点:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].tail(15).to_string())
    else:
        print(f"\n未检测到任何反转点")
    
    print(f"\n{'='*70}")
    print("PHASE1 完成!")
    print(f"{'='*70}")
    
    return labeled_df, reversals_df


if __name__ == '__main__':
    main()
