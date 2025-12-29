"""
Phase 2.5: Feature Sampling & Reversal Pattern Extraction
=========================================================
目標：從已標註的反轉點中提取特徵樣本，為模型訓練準備數據集

輸入：features_data.csv（包含所有技術指標）
輸出：reversal_samples.csv（每個反轉點的特徵快照）
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class ReversalSampler:
    """
    從反轉點提取特徵樣本的類
    
    策略：
    1. 識別所有反轉點（Label=1）
    2. 對每個反轉點提取前後 N 根 K 線的特徵
    3. 計算反轉強度（未來漲跌幅）
    4. 平衡正負樣本
    """
    
    def __init__(self, 
                 features_df: pd.DataFrame,
                 lookback_bars: int = 5,
                 lookahead_bars: int = 5,
                 min_samples: int = 10):
        """
        參數：
        - features_df: 包含特徵和標籤的 DataFrame
        - lookback_bars: 反轉前看回多少根 K 線
        - lookahead_bars: 反轉後看前多少根 K 線確認強度
        - min_samples: 最少需要多少個樣本
        """
        self.df = features_df.copy()
        self.lookback_bars = lookback_bars
        self.lookahead_bars = lookahead_bars
        self.min_samples = min_samples
        
        # 特徵列（除了 OHLCV 和標籤之外）
        self.feature_cols = [col for col in self.df.columns 
                            if col not in ['timestamp', 'open', 'high', 'low', 
                                          'close', 'volume', 'confirmed_label',
                                          'swing_type', 'raw_label', 
                                          'future_move_pct', 'is_confirmed_reversal']]
    
    def _calculate_reversal_strength(self, 
                                     idx: int, 
                                     reversal_type: str) -> float:
        """
        計算反轉的強度（未來漲跌幅）
        
        reversal_type: 'high' 或 'low'
        """
        if idx + self.lookahead_bars >= len(self.df):
            return np.nan
        
        current_close = self.df.iloc[idx]['close']
        future_prices = self.df.iloc[idx:idx + self.lookahead_bars + 1]['close'].values
        
        if reversal_type == 'high':
            # 反轉頂，未來應該下跌
            min_future = future_prices.min()
            strength = (current_close - min_future) / current_close * 100
        else:
            # 反轉底，未來應該上升
            max_future = future_prices.max()
            strength = (max_future - current_close) / current_close * 100
        
        return strength
    
    def extract_reversal_samples(self) -> pd.DataFrame:
        """
        提取所有反轉點的特徵樣本
        
        返回：DataFrame，每一行是一個反轉樣本
        """
        samples = []
        
        # 找出所有反轉點
        reversal_indices = self.df[self.df['confirmed_label'] == 1].index.tolist()
        
        if len(reversal_indices) == 0:
            print("警告：沒有找到反轉點！")
            return pd.DataFrame()
        
        print(f"找到 {len(reversal_indices)} 個反轉點")
        
        for idx in reversal_indices:
            # 檢查是否有足夠的前後數據
            if idx < self.lookback_bars or idx + self.lookahead_bars >= len(self.df):
                continue
            
            # 提取反轉前 N 根 K 線的特徵
            sample_df = self.df.iloc[idx - self.lookback_bars:idx + 1]
            
            # 建立樣本
            sample = {
                'sample_id': f"REV_{idx}",
                'candle_index': idx,
                'timestamp': self.df.iloc[idx]['timestamp'],
                'swing_type': self.df.iloc[idx]['swing_type'],
            }
            
            # 添加當前 K 線的特徵
            for col in self.feature_cols:
                sample[f'{col}_current'] = self.df.iloc[idx][col]
            
            # 添加前 N 根的平均特徵
            for col in self.feature_cols:
                lookback_values = sample_df[col].dropna().values
                if len(lookback_values) > 0:
                    sample[f'{col}_mean'] = lookback_values.mean()
                    sample[f'{col}_max'] = lookback_values.max()
                    sample[f'{col}_min'] = lookback_values.min()
                    sample[f'{col}_std'] = lookback_values.std() if len(lookback_values) > 1 else 0
                else:
                    sample[f'{col}_mean'] = np.nan
                    sample[f'{col}_max'] = np.nan
                    sample[f'{col}_min'] = np.nan
                    sample[f'{col}_std'] = np.nan
            
            # 計算反轉強度
            reversal_type = 'high' if self.df.iloc[idx]['swing_type'] == 'top' else 'low'
            strength = self._calculate_reversal_strength(idx, reversal_type)
            sample['reversal_strength'] = strength
            
            # 添加 OHLCV 信息
            sample['open'] = self.df.iloc[idx]['open']
            sample['high'] = self.df.iloc[idx]['high']
            sample['low'] = self.df.iloc[idx]['low']
            sample['close'] = self.df.iloc[idx]['close']
            sample['volume'] = self.df.iloc[idx]['volume']
            
            # 標籤
            sample['is_reversal'] = 1
            
            samples.append(sample)
        
        print(f"提取了 {len(samples)} 個有效反轉樣本")
        
        return pd.DataFrame(samples)
    
    def extract_non_reversal_samples(self, ratio: float = 1.0) -> pd.DataFrame:
        """
        提取非反轉點樣本（用於負例訓練）
        
        ratio: 非反轉樣本相對於反轉樣本的比例
        """
        samples = []
        reversal_indices = set(self.df[self.df['confirmed_label'] == 1].index.tolist())
        
        # 找出非反轉點
        non_reversal_indices = [i for i in range(len(self.df)) 
                                if i not in reversal_indices]
        
        # 隨機採樣（避免樣本不平衡太嚴重）
        num_samples = int(len(reversal_indices) * ratio)
        selected_indices = np.random.choice(non_reversal_indices, 
                                           min(num_samples, len(non_reversal_indices)),
                                           replace=False)
        
        print(f"採樣 {len(selected_indices)} 個非反轉樣本")
        
        for idx in selected_indices:
            if idx < self.lookback_bars or idx + self.lookahead_bars >= len(self.df):
                continue
            
            sample_df = self.df.iloc[idx - self.lookback_bars:idx + 1]
            
            sample = {
                'sample_id': f"NON_REV_{idx}",
                'candle_index': idx,
                'timestamp': self.df.iloc[idx]['timestamp'],
                'swing_type': 'none',
            }
            
            # 添加當前 K 線的特徵
            for col in self.feature_cols:
                sample[f'{col}_current'] = self.df.iloc[idx][col]
            
            # 添加前 N 根的統計特徵
            for col in self.feature_cols:
                lookback_values = sample_df[col].dropna().values
                if len(lookback_values) > 0:
                    sample[f'{col}_mean'] = lookback_values.mean()
                    sample[f'{col}_max'] = lookback_values.max()
                    sample[f'{col}_min'] = lookback_values.min()
                    sample[f'{col}_std'] = lookback_values.std() if len(lookback_values) > 1 else 0
                else:
                    sample[f'{col}_mean'] = np.nan
                    sample[f'{col}_max'] = np.nan
                    sample[f'{col}_min'] = np.nan
                    sample[f'{col}_std'] = np.nan
            
            sample['reversal_strength'] = np.nan
            
            sample['open'] = self.df.iloc[idx]['open']
            sample['high'] = self.df.iloc[idx]['high']
            sample['low'] = self.df.iloc[idx]['low']
            sample['close'] = self.df.iloc[idx]['close']
            sample['volume'] = self.df.iloc[idx]['volume']
            
            sample['is_reversal'] = 0
            
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def create_balanced_dataset(self, 
                                negative_ratio: float = 2.0) -> pd.DataFrame:
        """
        建立平衡的訓練數據集
        
        返回：包含反轉和非反轉樣本的合併 DataFrame
        """
        # 提取反轉樣本
        reversal_samples = self.extract_reversal_samples()
        
        if len(reversal_samples) == 0:
            raise ValueError("沒有足夠的反轉樣本！")
        
        # 提取非反轉樣本（按比例）
        non_reversal_samples = self.extract_non_reversal_samples(
            ratio=negative_ratio
        )
        
        # 合併
        all_samples = pd.concat([reversal_samples, non_reversal_samples], 
                               ignore_index=True)
        
        # 洗牌
        all_samples = all_samples.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        print(f"\n最終數據集統計：")
        print(f"  總樣本數: {len(all_samples)}")
        print(f"  反轉樣本: {len(reversal_samples)} ({len(reversal_samples)/len(all_samples)*100:.1f}%)")
        print(f"  非反轉樣本: {len(non_reversal_samples)} ({len(non_reversal_samples)/len(all_samples)*100:.1f}%)")
        print(f"  正負比例: 1:{negative_ratio}")
        
        return all_samples
    
    def get_feature_statistics(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """
        計算特徵統計（用於特徵工程分析）
        """
        stats = []
        
        for col in self.feature_cols:
            current_col = f'{col}_current'
            if current_col in samples_df.columns:
                stats.append({
                    'feature': col,
                    'mean': samples_df[current_col].mean(),
                    'std': samples_df[current_col].std(),
                    'min': samples_df[current_col].min(),
                    'max': samples_df[current_col].max(),
                    'missing': samples_df[current_col].isna().sum(),
                })
        
        return pd.DataFrame(stats)


def main():
    """
    主函數：執行特徵採樣
    """
    import sys
    
    # 參數
    pair = "BTCUSDT"
    timeframe = "15m"
    lookback_bars = 5
    lookahead_bars = 5
    negative_ratio = 2.0  # 非反轉樣本 / 反轉樣本比例
    
    # 文件路徑
    input_file = f"{pair}_{timeframe}_features.csv"
    output_file = f"{pair}_{timeframe}_samples.csv"
    stats_file = f"{pair}_{timeframe}_feature_stats.csv"
    
    print(f"Phase 2.5: 特徵採樣")
    print("=" * 60)
    print(f"配對: {pair}")
    print(f"時間框架: {timeframe}")
    print(f"回看期: {lookback_bars} 根 K 線")
    print(f"前看期: {lookahead_bars} 根 K 線")
    print(f"非反轉比例: {negative_ratio}:1")
    print("=" * 60)
    
    # 加載數據
    print(f"\n加載數據: {input_file}")
    try:
        features_df = pd.read_csv(input_file)
        print(f"  已加載 {len(features_df)} 行數據")
    except FileNotFoundError:
        print(f"錯誤：找不到文件 {input_file}")
        sys.exit(1)
    
    # 初始化採樣器
    sampler = ReversalSampler(features_df, 
                              lookback_bars=lookback_bars,
                              lookahead_bars=lookahead_bars)
    
    # 建立平衡數據集
    print(f"\n提取特徵樣本...")
    samples_df = sampler.create_balanced_dataset(negative_ratio=negative_ratio)
    
    # 保存樣本
    print(f"\n保存樣本: {output_file}")
    samples_df.to_csv(output_file, index=False)
    print(f"  ✓ 完成")
    
    # 計算特徵統計
    print(f"\n計算特徵統計...")
    stats_df = sampler.get_feature_statistics(samples_df)
    stats_df.to_csv(stats_file, index=False)
    print(f"  ✓ 已保存: {stats_file}")
    
    # 打印樣本信息
    print(f"\n樣本詳情:")
    print(f"  反轉樣本（標籤=1）:")
    reversal_count = (samples_df['is_reversal'] == 1).sum()
    print(f"    數量: {reversal_count}")
    if reversal_count > 0:
        print(f"    反轉強度平均值: {samples_df[samples_df['is_reversal']==1]['reversal_strength'].mean():.2f}%")
    
    print(f"  非反轉樣本（標籤=0）: {(samples_df['is_reversal'] == 0).sum()}")
    
    # 打印前 10 行樣本
    print(f"\n前 10 個樣本:")
    display_cols = ['sample_id', 'timestamp', 'swing_type', 
                    'rsi_14_current', 'bb_percent_b_current', 
                    'volume_spike_current', 'reversal_strength', 'is_reversal']
    available_cols = [col for col in display_cols if col in samples_df.columns]
    print(samples_df[available_cols].head(10).to_string())
    
    print(f"\n✓ Phase 2.5 完成！")
    print(f"  輸出文件: {output_file}")


if __name__ == "__main__":
    main()
