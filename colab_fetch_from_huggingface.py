#!/usr/bin/env python3
"""
Colab脚本: 从HuggingFace Datasets抓取K线数据
功能: 自动下载、合并和分析Binance US K线数据
数据来源: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/klines_binance_us
"""

import os
import json
import pandas as pd
import requests
from pathlib import Path
import subprocess
import sys

# HuggingFace数据集配置
HF_DATASET_OWNER = "zongowo111"
HF_DATASET_NAME = "cpb-models"
HF_REPO_TYPE = "dataset"
HF_DATA_PATH = "klines_binance_us"

# 支持的交易对
TRADING_PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "MATICUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "BCHUSDT",
    "ETCUSDT",
    "XLUSDT",
    "DOTUSDT",
    "TRXUSDT"
]

# 支持的时间周期
TIMEFRAMES = ["15m", "1h"]


class HuggingFaceKlinesDownloader:
    """从HuggingFace Datasets下载K线数据的类"""
    
    def __init__(self):
        self.hf_dataset = f"{HF_DATASET_OWNER}/{HF_DATASET_NAME}"
        self.data_path = HF_DATA_PATH
        self.output_dir = "downloaded_klines"
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def install_dependencies(self):
        """安装必需的依赖"""
        print("安装必需的库...")
        packages = ["datasets", "huggingface-hub", "pandas", "numpy"]
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("依赖安装完成")
    
    def download_klines_from_hf(self, trading_pair, timeframe="15m"):
        """从HuggingFace下载指定交易对的K线数据
        
        Args:
            trading_pair: 交易对，例如 "BTCUSDT"
            timeframe: 时间周期，"15m" 或 "1h"
            
        Returns:
            DataFrame: K线数据
        """
        try:
            print(f"下载 {trading_pair} {timeframe} K线数据...")
            
            # 使用HuggingFace Hub直接下载
            from huggingface_hub import hf_hub_download
            
            # 构建文件路径
            file_name = f"{trading_pair}_{timeframe}.csv"
            repo_id = f"{HF_DATASET_OWNER}/{HF_DATASET_NAME}"
            
            try:
                # 尝试下载CSV文件
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{HF_DATA_PATH}/{file_name}",
                    repo_type=HF_REPO_TYPE
                )
                
                # 读取CSV
                df = pd.read_csv(file_path)
                print(f"成功下载 {trading_pair} {timeframe}: {len(df)} 行数据")
                return df
                
            except Exception as e:
                print(f"直接下载失败: {e}")
                print(f"尝试使用datasets库...")
                
                # 备选方法：使用datasets库
                from datasets import load_dataset
                dataset = load_dataset(f"{HF_DATASET_OWNER}/cpb-models", split="train")
                
                # 过滤指定的交易对和时间周期
                filtered = dataset.filter(
                    lambda x: x.get("pair") == trading_pair and x.get("timeframe") == timeframe
                )
                
                if len(filtered) > 0:
                    df = filtered.to_pandas()
                    print(f"成功加载 {trading_pair} {timeframe}: {len(df)} 行数据")
                    return df
                else:
                    return None
                    
        except Exception as e:
            print(f"下载 {trading_pair} {timeframe} 失败: {e}")
            return None
    
    def fetch_and_combine(self, trading_pair, timeframe="15m", rows=10000):
        """获取并整合K线数据
        
        Args:
            trading_pair: 交易对
            timeframe: 时间周期
            rows: 返回的行数
            
        Returns:
            DataFrame: 整合后的K线数据
        """
        df = self.download_klines_from_hf(trading_pair, timeframe)
        
        if df is not None:
            # 确保列名正确
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            
            # 检查列名
            if all(col in df.columns for col in required_columns):
                # 保留必需列
                df = df[required_columns]
            elif all(col.lower() in [c.lower() for c in df.columns] for col in required_columns):
                # 列名不区分大小写，重新命名
                rename_dict = {}
                for col in df.columns:
                    for req_col in required_columns:
                        if col.lower() == req_col.lower():
                            rename_dict[col] = req_col
                df = df.rename(columns=rename_dict)
                df = df[required_columns]
            else:
                print(f"警告: 数据格式不符。现有列: {df.columns.tolist()}")
                return None
            
            # 取最后N行（最新的数据）
            df = df.tail(rows).reset_index(drop=True)
            
            # 确保timestamp是字符串格式
            df["timestamp"] = df["timestamp"].astype(str)
            
            # 保存到本地
            output_file = f"{self.output_dir}/{trading_pair}_{timeframe}_{rows}.csv"
            df.to_csv(output_file, index=False)
            print(f"已保存到: {output_file}")
            
            return df
        else:
            return None
    
    def list_available_data(self):
        """列出HuggingFace上可用的数据"""
        print("可用的数据列表:")
        print("交易对列表:")
        for pair in TRADING_PAIRS:
            print(f"  - {pair}")
        print("\n支持的时间周期:")
        for tf in TIMEFRAMES:
            print(f"  - {tf}")
        
        print(f"\n数据来源: https://huggingface.co/datasets/{self.hf_dataset}/tree/main/{self.data_path}")


def main():
    """主函数：执行数据下载流程"""
    
    print("="*60)
    print("HuggingFace K线数据下载工具")
    print("="*60)
    
    # 初始化下载器
    downloader = HuggingFaceKlinesDownloader()
    
    # 安装依赖
    downloader.install_dependencies()
    
    # 列出可用数据
    downloader.list_available_data()
    
    # 下载示例数据
    print("\n")
    print("开始下载K线数据...")
    print("="*60)
    
    # 下载BTC USDT 15分钟数据
    df_btc = downloader.fetch_and_combine(
        trading_pair="BTCUSDT",
        timeframe="15m",
        rows=10000
    )
    
    if df_btc is not None:
        print(f"\nBTC USDT 数据统计:")
        print(f"  行数: {len(df_btc)}")
        print(f"  列数: {len(df_btc.columns)}")
        print(f"  日期范围: {df_btc['timestamp'].iloc[0]} 到 {df_btc['timestamp'].iloc[-1]}")
        print(f"\n前5行数据:")
        print(df_btc.head())
        print(f"\n数据已保存，可以用于后续分析")
    
    return df_btc


if __name__ == "__main__":
    df = main()
