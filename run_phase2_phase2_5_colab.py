"""
Colab 執行腳本：Phase 2 + Phase 2.5 完整流程
================================================
執行步驟：
1. Clone 倉庫
2. 從 HuggingFace 下載 BTC 15m 數據
3. 計算特徵 (Phase 2)
4. 提取反轉樣本 (Phase 2.5)
5. 下載成果
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# STEP 1: Clone 倉庫
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Clone 倉庫")
print("="*70)

if not os.path.exists('swing-reversal-prediction'):
    os.system('git clone https://github.com/caizongxun/swing-reversal-prediction.git')
else:
    print("✓ 倉庫已經存在，正在更新...")
    os.system('cd swing-reversal-prediction && git pull')

os.chdir('swing-reversal-prediction')
print("✓ 這動完成")

# ============================================================================
# STEP 2: 安裝依賴
# ============================================================================
print("\n" + "="*70)
print("STEP 2: 安裝依賴")
print("="*70)

os.system('pip install huggingface-hub tqdm -q')
print("✓ 依賴安裝完成")

# ============================================================================
# STEP 3: 下載數據
# ============================================================================
print("\n" + "="*70)
print("STEP 3: 從 HuggingFace 下載 BTC 15m 數據")
print("="*70)

from huggingface_hub import hf_hub_download

REPO_ID = "zongowo111/cpb-models"
PAIR = "BTCUSDT"
TIMEFRAME = "15m"

filename = f"{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv"

print(f"下載: {PAIR} {TIMEFRAME}...")

try:
    csv_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir="./data",
        force_download=False
    )
    print(f"✓ 下載完成: {csv_path}")
    
    # 載入數據
    df_raw = pd.read_csv(csv_path)
    print(f"  數據大小: {df_raw.shape}")
    print(f"  時間範圍: {df_raw.iloc[0][0]} 至 {df_raw.iloc[-1][0]}")
    
except Exception as e:
    print(f"⌫ 下載失敗: {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: Phase 2 - 特徵工程
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Phase 2 - 特徵工程")
print("="*70)

from feature_engineering import ReversalFeatureEngineer

# 準備數據 (OHLCV + 標籤列)
df_prepared = df_raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

# 添加標籤列 (從 Phase 1 的 labeled_data.csv 程式上佐)
if 'confirmed_label' not in df_prepared.columns:
    df_prepared['confirmed_label'] = 0
    df_prepared['swing_type'] = ''
    df_prepared['raw_label'] = 0
    df_prepared['future_move_pct'] = 0.0
    df_prepared['is_confirmed_reversal'] = False

print("計算特徵...")
engineering = ReversalFeatureEngineer(df_prepared)
features_df = engineering.compute_all_features()

print(f"\n✓ Phase 2 完成")
print(f"  輸出: {len(features_df)} 行 × {len(features_df.columns)} 列")
print(f"  新特徵數: 16")

# 保存特徵數據
features_file = f"{PAIR}_{TIMEFRAME}_features.csv"
features_df.to_csv(features_file, index=False)
print(f"  保存: {features_file}")

# 打印樣本
print(f"\n前 5 行樣本:")
display_cols = [col for col in features_df.columns[:15]]
print(features_df[display_cols].head())

# ============================================================================
# STEP 5: Phase 2.5 - 特徵採樣
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Phase 2.5 - 特徵採樣")
print("="*70)

from feature_sampling import ReversalSampler

print("執行特徵採樣...")

sampler = ReversalSampler(
    features_df,
    lookback_bars=5,      # 反轉前 5 根 K 線
    lookahead_bars=5,     # 反轉後 5 根 K 線確認強度
    min_samples=10
)

# 建立平衡數據集
samples_df = sampler.create_balanced_dataset(negative_ratio=2.0)

print(f"\n✓ Phase 2.5 完成")
print(f"  輸出: {len(samples_df)} 行")

# 保存樣本
samples_file = f"{PAIR}_{TIMEFRAME}_samples.csv"
samples_df.to_csv(samples_file, index=False)
print(f"  保存: {samples_file}")

# 計算特徵統計
stats_df = sampler.get_feature_statistics(samples_df)
stats_file = f"{PAIR}_{TIMEFRAME}_feature_stats.csv"
stats_df.to_csv(stats_file, index=False)
print(f"  特徵統計: {stats_file}")

# 打印前 10 個樣本
print(f"\n前 10 個樣本:")
display_cols = ['sample_id', 'timestamp', 'swing_type', 
                'rsi_14_current', 'bb_percent_b_current', 'is_reversal']
available_cols = [col for col in display_cols if col in samples_df.columns]
print(samples_df[available_cols].head(10).to_string())

# ============================================================================
# STEP 6: 下載成果
# ============================================================================
print("\n" + "="*70)
print("STEP 6: 下載成果")
print("="*70)

try:
    from google.colab import files
    
    print(f"正在準備下載...")
    files.download(features_file)
    print(f"✓ 時間下載: {features_file}")
    
    files.download(samples_file)
    print(f"✓ 樣本下載: {samples_file}")
    
    files.download(stats_file)
    print(f"✓ 統計下載: {stats_file}")
    
except ImportError:
    print("⚠ 不是 Colab 環境，樣本洗保存在當前索引上:")
    print(f"  {features_file}")
    print(f"  {samples_file}")
    print(f"  {stats_file}")

# ============================================================================
# FINAL
# ============================================================================
print("\n" + "="*70)
print("✅ Phase 2 + Phase 2.5 完成！")
print("="*70)
print(f"\
下一步：Phase 3 - 特徵探索 & 模型訓練
- 計算特徵重要性 (Importance)
- 發現自動公式 (符號回歸)
- 訓練特後一獲收被：Random Forest / Decision Tree
")
