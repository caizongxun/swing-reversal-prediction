#!/usr/bin/env python3
"""
Colab 脚本: 最终版本 - 从 HuggingFace Datasets 下载 K 线数据
功能: 自动下载、合并和分析 Binance US K 线数据
数据来源: https://huggingface.co/datasets/zongowo111/cpb-models

最终正确路径:
klines_binance_us/{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv

例项:
https://huggingface.co/datasets/zongowo111/cpb-models/blob/main/klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv
"""

import os
import sys
import json
import pandas as pd
import subprocess
from pathlib import Path

# HuggingFace 数据集配置
HF_DATASET_OWNER = "zongowo111"
HF_DATASET_NAME = "cpb-models"
HF_REPO_TYPE = "dataset"

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
    "XLMUSDT",
    "VETUSDT",
    "FILUSDT",
    "THETAUSDT",
    "NEARUSDT",
    "APEUSDT"
]

# 支持的时间周期
TIMEFRAMES = ["15m", "1h"]


class HuggingFaceKlinesDownloader:
    """从 HuggingFace Datasets 下载 K 线数据的类"""
    
    def __init__(self):
        self.hf_dataset = f"{HF_DATASET_OWNER}/{HF_DATASET_NAME}"
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
        """从 HuggingFace 下载指定交易对的 K 线数据
        
        最终正确路径:
        klines_binance_us/{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv
        
        例项:
        - BTCUSDT 15m: klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv
        - ETHUSDT 1h:  klines_binance_us/ETHUSDT/ETHUSDT_1h_binance_us.csv
        
        Args:
            trading_pair: 交易对，例如 "BTCUSDT"
            timeframe: 时间周期，"15m" 或 "1h"
            
        Returns:
            DataFrame: K 线数据
        """
        try:
            print(f"下载 {trading_pair} {timeframe} K 线数据...")
            
            from huggingface_hub import hf_hub_download
            
            # 最终正确的文件路径
            file_name = f"{trading_pair}_{timeframe}_binance_us.csv"
            repo_id = f"{HF_DATASET_OWNER}/{HF_DATASET_NAME}"
            file_path_in_repo = f"klines_binance_us/{trading_pair}/{file_name}"
            
            print(f"  执行的路径: {file_path_in_repo}")
            
            try:
                # 下载 CSV 文件
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path_in_repo,
                    repo_type=HF_REPO_TYPE
                )
                
                # 读取 CSV
                df = pd.read_csv(file_path)
                print(f"  ✓ 成功下载: {len(df)} 行数据")
                return df
                
            except Exception as e:
                print(f"  ✗ 直接下载失败: {str(e)[:100]}")
                return None
                    
        except Exception as e:
            print(f"  ✗ 错误: {str(e)[:100]}")
            return None
    
    def fetch_and_combine(self, trading_pair, timeframe="15m", rows=10000):
        """获取并整合 K 线数据
        
        Args:
            trading_pair: 交易对
            timeframe: 时间周期
            rows: 返回的行数
            
        Returns:
            DataFrame: 整合后的 K 线数据
        """
        df = self.download_klines_from_hf(trading_pair, timeframe)
        
        if df is not None:
            # 确保列名正确
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            
            # 检查列名
            print(f"  数据列名: {df.columns.tolist()}")
            
            # 处理列名活性
            actual_cols = [col.lower().strip() for col in df.columns]
            if all(req.lower() in actual_cols for req in required_columns):
                # 列名不区分大小写，重新映射
                col_mapping = {}
                for actual_col in df.columns:
                    for req_col in required_columns:
                        if actual_col.lower().strip() == req_col.lower():
                            col_mapping[actual_col] = req_col
                df = df.rename(columns=col_mapping)
                df = df[required_columns]
            elif all(col in df.columns for col in required_columns):
                df = df[required_columns]
            else:
                print(f"  警告: 列名预期不符")
                print(f"  需要: {required_columns}")
                print(f"  实际: {df.columns.tolist()}")
                return None
            
            # 取最后 N 行（最新的数据）
            df = df.tail(rows).reset_index(drop=True)
            
            # 确保 timestamp 是字符串格式
            df["timestamp"] = df["timestamp"].astype(str)
            
            # 保存到本地
            output_file = f"{self.output_dir}/{trading_pair}_{timeframe}_{rows}.csv"
            df.to_csv(output_file, index=False)
            print(f"  保存到: {output_file}")
            
            return df
        else:
            return None
    
    def list_available_data(self):
        """列出 HuggingFace 上可用的数据"""
        print("可用的数据列表:")
        print("\n交易对列表:")
        for pair in TRADING_PAIRS:
            print(f"  - {pair}")
        print("\n支持的时间周期:")
        for tf in TIMEFRAMES:
            print(f"  - {tf}")
        
        print(f"\n数据来源: https://huggingface.co/datasets/{self.hf_dataset}")
        print(f"正确路径: klines_binance_us/{{PAIR}}/{{PAIR}}_{{TIMEFRAME}}_binance_us.csv")
        print(f"\n示例 URL: ")
        print(f"https://huggingface.co/datasets/zongowo111/cpb-models/blob/main/klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv")


def main():
    """主函数：执行数据下载流程"""
    
    print("="*60)
    print("HuggingFace K 线数据下载工具 (最终版)")
    print("="*60)
    
    # 初始化下载器
    downloader = HuggingFaceKlinesDownloader()
    
    # 安装依赖
    downloader.install_dependencies()
    
    # 列出可用数据
    downloader.list_available_data()
    
    # 下载示例数据
    print("\n")
    print("开始下载 K 线数据...")
    print("="*60)
    
    # 下载 BTC USDT 15 分钟数据
    df_btc = downloader.fetch_and_combine(
        trading_pair="BTCUSDT",
        timeframe="15m",
        rows=10000
    )
    
    if df_btc is not None:
        print(f"\n✓ 成功！")
        print(f"\nBTC USDT 数据统计:")
        print(f"  行数: {len(df_btc)}")
        print(f"  列数: {len(df_btc.columns)}")
        print(f"  日期范围: {df_btc['timestamp'].iloc[0]} 到 {df_btc['timestamp'].iloc[-1]}")
        print(f"\n前 5 行数据:")
        print(df_btc.head())
        print(f"\n✓ 数据已保存，可以用于后续分析")
    else:
        print(f"\n✗ 下载失败，请检查以上错误信息")
    
    return df_btc


if __name__ == "__main__":
    df = main()
