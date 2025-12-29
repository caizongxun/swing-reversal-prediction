#!/usr/bin/env python3
"""
Colab完整管道脚本: 从HuggingFace下载K线数据，然后执行反轉检测分析
步骤:
  1. 从HuggingFace Datasets下载K线数据
  2. 执行5阶段反轉检测分析
  3. 导出结果文件
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

# HuggingFace配置
HF_DATASET = "zongowo111/cpb-models"
HF_DATA_PATH = "klines_binance_us"
HF_TRADING_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "MATICUSDT", "AVAXUSDT", "LINKUSDT"
]
HF_TIMEFRAMES = ["15m", "1h"]


class HFKlinesDownloader:
    """从HuggingFace下载K线数据"""
    
    def __init__(self):
        self.output_dir = "downloaded_klines"
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def install_packages(self):
        """安装必需包"""
        print("安装依赖库...")
        packages = ["datasets", "huggingface-hub"]
        for pkg in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        print("依赖库安装完成")
    
    def download_klines(self, trading_pair, timeframe="15m", rows=10000):
        """从HuggingFace下载指定交易对的K线数据
        
        Args:
            trading_pair: 例如 "BTCUSDT"
            timeframe: "15m" 或 "1h"
            rows: 返回行数
            
        Returns:
            DataFrame 或 None
        """
        try:
            print(f"下载 {trading_pair} {timeframe}...")
            
            from huggingface_hub import hf_hub_download
            
            file_name = f"{trading_pair}_{timeframe}.csv"
            repo_id = HF_DATASET
            
            # 尝试直接下载CSV
            try:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{HF_DATA_PATH}/{file_name}",
                    repo_type="dataset"
                )
                
                df = pd.read_csv(file_path)
                print(f"成功: {len(df)} 行数据")
                
                # 验证列
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                if all(col in df.columns for col in required_cols):
                    df = df[required_cols]
                    df = df.tail(rows).reset_index(drop=True)
                    return df
                else:
                    print(f"列名检查失败。现有列: {df.columns.tolist()}")
                    return None
                    
            except Exception as e:
                print(f"直接下载失败: {e}")
                return None
                
        except Exception as e:
            print(f"下载过程出错: {e}")
            return None
    
    def interactive_download(self):
        """互动式下载单个交易对数据"""
        print("\n可用的交易对:")
        for i, pair in enumerate(HF_TRADING_PAIRS, 1):
            print(f"  {i}. {pair}")
        
        pair_index = input(f"\n选择交易对 (1-{len(HF_TRADING_PAIRS)}, 默认 1): ").strip()
        pair_index = int(pair_index) if pair_index else 1
        selected_pair = HF_TRADING_PAIRS[pair_index - 1]
        
        print("\n可用的时间周期:")
        for i, tf in enumerate(HF_TIMEFRAMES, 1):
            print(f"  {i}. {tf}")
        
        tf_index = input(f"\n选择时间周期 (1-{len(HF_TIMEFRAMES)}, 默认 1): ").strip()
        tf_index = int(tf_index) if tf_index else 1
        selected_tf = HF_TIMEFRAMES[tf_index - 1]
        
        rows = input("\n输入K线数量 (默认 10000): ").strip()
        rows = int(rows) if rows else 10000
        
        print(f"\n开始下载 {selected_pair} {selected_tf} ({rows}行)...\n")
        return self.download_klines(selected_pair, selected_tf, rows)


class SwingReversalAnalyzer:
    """反轉检测分析器"""
    
    def __init__(self, window=5, future_candles=12, threshold=1.0):
        self.window = window
        self.future_candles = future_candles
        self.threshold = threshold
    
    def detect_swings(self, df):
        """检测振幅点"""
        print("Phase 1: 检测振幅点...")
        
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        
        swing_highs = []
        swing_lows = []
        
        for i in range(self.window, n - self.window):
            # 检测局部最高点
            if all(highs[i] > highs[j] for j in range(i - self.window, i + self.window + 1) if j != i):
                swing_highs.append(i)
            
            # 检测局部最低点
            if all(lows[i] < lows[j] for j in range(i - self.window, i + self.window + 1) if j != i):
                swing_lows.append(i)
        
        print(f"  检测到 {len(swing_highs)} 个局部最高点")
        print(f"  检测到 {len(swing_lows)} 个局部最低点")
        
        return swing_highs, swing_lows
    
    def confirm_reversals(self, df, swing_highs, swing_lows):
        """验证反轉"""
        print("\nPhase 2: 验证反轉...")
        
        confirmed = []
        closes = df["close"].values
        
        # 验证高点反轉
        for idx in swing_highs:
            if idx + self.future_candles < len(df):
                future_min = closes[idx + 1 : idx + self.future_candles + 1].min()
                change_pct = (closes[idx] - future_min) / closes[idx] * 100
                
                if change_pct >= self.threshold:
                    confirmed.append((idx, "HIGH", change_pct))
        
        # 验证低点反轉
        for idx in swing_lows:
            if idx + self.future_candles < len(df):
                future_max = closes[idx + 1 : idx + self.future_candles + 1].max()
                change_pct = (future_max - closes[idx]) / closes[idx] * 100
                
                if change_pct >= self.threshold:
                    confirmed.append((idx, "LOW", change_pct))
        
        print(f"  确认 {len(confirmed)} 个反轉点")
        return confirmed
    
    def extract_features(self, df, indices, window=20):
        """提取特征"""
        print("\nPhase 3: 提取特征...")
        
        features_list = []
        
        for idx, reversal_type, move_pct in indices:
            if idx >= window:
                window_df = df.iloc[idx - window : idx]
                
                features = {
                    "index": idx,
                    "type": reversal_type,
                    "move_pct": move_pct,
                    "volume_ratio": window_df["volume"].iloc[-1] / window_df["volume"].mean(),
                    "price_change": (window_df["close"].iloc[-1] - window_df["close"].iloc[0]) / window_df["close"].iloc[0] * 100,
                    "volatility": window_df["close"].std() / window_df["close"].mean(),
                }
                features_list.append(features)
        
        print(f"  提取 {len(features_list)} 组特征")
        return pd.DataFrame(features_list) if features_list else None
    
    def analyze(self, df, output_prefix="reversal"):
        """完整分析流程"""
        print("\n" + "="*60)
        print("开始反轉检测分析")
        print("="*60)
        
        # Phase 1: 检测
        swing_highs, swing_lows = self.detect_swings(df)
        
        # Phase 2: 确认
        confirmed = self.confirm_reversals(df, swing_highs, swing_lows)
        
        if not confirmed:
            print("\n未检测到符合条件的反轉点")
            return None
        
        # Phase 3: 提取特征
        features_df = self.extract_features(df, confirmed)
        
        # 导出结果
        print("\nPhase 4: 导出结果...")
        
        output_file = f"{output_prefix}_analysis_result.csv"
        if features_df is not None:
            features_df.to_csv(output_file, index=False)
            print(f"  已保存特征文件: {output_file}")
        
        # 保存标记数据
        df_labeled = df.copy()
        df_labeled["reversal_label"] = 0
        for idx, rtype, _ in confirmed:
            df_labeled.loc[idx, "reversal_label"] = 1
        
        labeled_file = f"{output_prefix}_labeled_data.csv"
        df_labeled.to_csv(labeled_file, index=False)
        print(f"  已保存标记数据: {labeled_file}")
        
        print("\n分析完成！")
        print("="*60)
        
        return {
            "features": features_df,
            "labeled_data": df_labeled,
            "confirmed_count": len(confirmed)
        }


def main():
    """主函数"""
    print("\n" + "#"*60)
    print("# HuggingFace K线数据 + 反轉检测完整管道")
    print("#"*60)
    
    # 步骤 1: 下载数据
    print("\n步骤 1: 从HuggingFace下载K线数据")
    print("-"*60)
    
    downloader = HFKlinesDownloader()
    downloader.install_packages()
    
    # 互动式选择或直接使用默认
    interactive = input("\n是否互动式选择数据? (y/n, 默认 n): ").strip().lower() == "y"
    
    if interactive:
        df = downloader.interactive_download()
    else:
        print("使用默认数据: BTCUSDT 15分钟")
        df = downloader.download_klines("BTCUSDT", "15m", 10000)
    
    if df is None:
        print("\n数据下载失败，程序退出")
        return
    
    print(f"\n下载完成: {len(df)} 行数据")
    print(f"日期范围: {df['timestamp'].iloc[0]} 至 {df['timestamp'].iloc[-1]}")
    
    # 步骤 2: 执行分析
    print("\n步骤 2: 执行反轉检测分析")
    print("-"*60)
    
    analyzer = SwingReversalAnalyzer(
        window=5,
        future_candles=12,
        threshold=1.0
    )
    
    results = analyzer.analyze(df, output_prefix="hf_klines")
    
    if results:
        print(f"\n分析统认:")
        print(f"  确认反轉点: {results['confirmed_count']}")
        if results['features'] is not None:
            print(f"  提取特征数: {len(results['features'])}")
    
    print("\n所有结果文件已生成")


if __name__ == "__main__":
    main()
