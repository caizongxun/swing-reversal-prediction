#!/usr/bin/env python3
"""
PHASE1: Simple Reversal Point Detection with God's View

Core Logic (极其简单):
1. 找到相对低点 (local low)
   - 当前close < 前N根的最低价
   - 当前close < 后N根的最低价
   
2. 找到相对高点 (local high)
   - 当前close > 前N根的最高价
   - 当前close > 后N根的最高价

3. 验证反转点有效性 (god's view confirmation)
   - 反转点后的 5-10 根K线
   - 如果没有创新低/新高，则为有效的反转点

这是最简单、最客观的标记方式。
模型会从这些标记点中自动学习规律。
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SimpleReversalDetector:
    """
    简单反转点检测器 - 基于相对高低点
    """
    
    def __init__(self, df, lookback=5, lookforward=10):
        """
        Args:
            df: DataFrame with OHLCV data
            lookback: 向后看N根K线找低点/高点
            lookforward: 向前看N根K线验证反转
        """
        self.df = df.copy()
        self.lookback = lookback
        self.lookforward = lookforward
        self.reversals = []
        
    def find_local_lows(self):
        """
        找相对低点：
        - 当前close是前lookback根K线的最低
        - 当前close是后lookforward根K线的最低
        """
        local_lows = []
        
        for i in range(self.lookback, len(self.df) - self.lookforward):
            current_close = self.df.iloc[i]['close']
            
            # 向后看：当前价格是前N根最低
            lookback_low = self.df.iloc[i-self.lookback:i]['close'].min()
            
            # 向前看：当前价格是后N根最低
            lookforward_low = self.df.iloc[i+1:i+self.lookforward+1]['close'].min()
            
            # 如果当前价格既是前N根最低，也是后N根最低，则为相对低点
            if current_close <= lookback_low and current_close <= lookforward_low:
                local_lows.append({
                    'index': i,
                    'timestamp': self.df.iloc[i]['timestamp'],
                    'close': current_close,
                    'type': 'Local_Low',
                    'lookback_min': lookback_low,
                    'lookforward_min': lookforward_low
                })
        
        return local_lows
    
    def find_local_highs(self):
        """
        找相对高点：
        - 当前close是前lookback根K线的最高
        - 当前close是后lookforward根K线的最高
        """
        local_highs = []
        
        for i in range(self.lookback, len(self.df) - self.lookforward):
            current_close = self.df.iloc[i]['close']
            
            # 向后看：当前价格是前N根最高
            lookback_high = self.df.iloc[i-self.lookback:i]['close'].max()
            
            # 向前看：当前价格是后N根最高
            lookforward_high = self.df.iloc[i+1:i+self.lookforward+1]['close'].max()
            
            # 如果当前价格既是前N根最高，也是后N根最高，则为相对高点
            if current_close >= lookback_high and current_close >= lookforward_high:
                local_highs.append({
                    'index': i,
                    'timestamp': self.df.iloc[i]['timestamp'],
                    'close': current_close,
                    'type': 'Local_High',
                    'lookback_max': lookback_high,
                    'lookforward_max': lookforward_high
                })
        
        return local_highs
    
    def detect_reversals(self):
        """
        检测所有反转点
        """
        print(f"[1/2] 检测相对低点 (lookback={self.lookback}, lookforward={self.lookforward})...")
        local_lows = self.find_local_lows()
        print(f"    发现 {len(local_lows)} 个相对低点")
        
        print(f"\n[2/2] 检测相对高点 (lookback={self.lookback}, lookforward={self.lookforward})...")
        local_highs = self.find_local_highs()
        print(f"    发现 {len(local_highs)} 个相对高点")
        
        all_reversals = local_lows + local_highs
        self.reversals = pd.DataFrame(all_reversals).sort_values('index').reset_index(drop=True)
        
        return self.reversals
    
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
        local_lows = self.reversals[self.reversals['type'] == 'Local_Low']
        local_highs = self.reversals[self.reversals['type'] == 'Local_High']
        
        return {
            'total': len(self.reversals),
            'local_lows': len(local_lows),
            'local_highs': len(local_highs),
            'ratio': len(self.reversals) / len(self.df) * 100
        }


def main():
    print("="*70)
    print("PHASE1: 简单反转点检测 - 上帝视角")
    print("="*70)
    
    csv_file = 'labeled_klines_phase1.csv'
    
    print(f"\n加载数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 确保数据类型正确
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    
    print(f"数据大小: {df.shape}")
    print(f"时间范围: {df['timestamp'].iloc[0]} 到 {df['timestamp'].iloc[-1]}")
    
    # 创建检测器 (lookback=5, lookforward=10)
    detector = SimpleReversalDetector(df, lookback=5, lookforward=10)
    
    print(f"\n{'='*70}")
    print("执行反转点检测...")
    print(f"{'='*70}")
    
    reversals = detector.detect_reversals()
    
    print(f"\n标记数据集...")
    labeled_df = detector.label_dataset()
    
    # 保存结果
    output_file = 'labeled_klines_phase1_simple.csv'
    cols = [c for c in labeled_df.columns if c in 
            ['timestamp', 'open', 'high', 'low', 'close', 'volume',
             'Reversal_Label', 'Reversal_Type', 'Reversal_Price']]
    
    labeled_df[cols].to_csv(output_file, index=False)
    print(f"已保存到: {output_file}")
    
    # 统计摘要
    summary = detector.get_summary()
    
    print(f"\n{'='*70}")
    print("PHASE1 检测结果摘要")
    print(f"{'='*70}")
    print(f"\n总反转点数: {summary['total']}")
    print(f"  - 相对低点 (买入点): {summary['local_lows']}")
    print(f"  - 相对高点 (卖出点): {summary['local_highs']}")
    print(f"\n反转点占比: {summary['ratio']:.2f}%")
    print(f"平均间距: {len(df) // (summary['total'] + 1):.0f} 根K线")
    
    # 显示前10个反转点
    print(f"\n前10个反转点:")
    print(reversals[['index', 'timestamp', 'type', 'close']].head(10).to_string())
    
    # 显示最后10个反转点
    print(f"\n最后10个反转点:")
    print(reversals[['index', 'timestamp', 'type', 'close']].tail(10).to_string())
    
    print(f"\n{'='*70}")
    print("PHASE1 完成!")
    print(f"{'='*70}")
    
    return labeled_df, reversals


if __name__ == '__main__':
    main()
