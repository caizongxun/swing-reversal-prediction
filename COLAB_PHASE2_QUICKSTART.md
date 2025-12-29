# Phase 2 in Google Colab - Quick Start Guide

## 5-Minute Setup

### Step 1: Clone Repository

在 Colab Cell 中執行：

```bash
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
```

### Step 2: Upload Your Data

```python
from google.colab import files

print("Uploading labeled_data.csv...")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Uploaded: {filename}")
```

會彈出檔案上傳對話框，選擇你的 `labeled_data.csv`

### Step 3: Run Feature Engineering

```python
# 安裝依賴（Colab已預裝pandas, numpy）
import pandas as pd
import numpy as np
from feature_engineering import ReversalFeatureEngineer

# 載入數據
print("Loading data...")
df = pd.read_csv('labeled_data.csv')
print(f"Loaded {len(df)} rows")

# 初始化特徵工程師
engine = ReversalFeatureEngineer(df)

# 計算所有特徵
print("Computing features (this may take 1-2 minutes)...")
features_df = engine.compute_all_features()

print("Done!")
```

### Step 4: 查看結果

```python
# 顯示新特徵
new_features = [col for col in features_df.columns if col not in df.columns]
print(f"\nFeatures created: {len(new_features)}")
print(new_features)

# 顯示數據樣本
print("\nSample data (first 5 rows):")
display_cols = ['timestamp', 'close', 'volume', 'confirmed_label', 
                'rsi_14', 'bb_percent_b', 'hammer', 'volume_spike']
print(features_df[display_cols].head())
```

### Step 5: 分析特徵分佈

```python
# 確認反轉 vs 假信號的特徵對比
confirmed = features_df[features_df['confirmed_label'] == 1]
false_sig = features_df[features_df['confirmed_label'] == 0]

print("\nFeature Statistics Comparison:")
print("\nConfirmed Reversals (mean):")
print(confirmed[['rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum']].mean())

print("\nFalse Signals (mean):")
print(false_sig[['rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum']].mean())
```

### Step 6: 導出結果

```python
# 保存特徵數據
features_df.to_csv('features_data.csv', index=False)
print("Saved to features_data.csv")

# 下載文件
files.download('features_data.csv')
print("Download complete!")
```

---

## 完整 Colab 筆記本

### 所有代碼組合在一起

```python
# ============ CELL 1: 克隆倉庫 ============
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
!ls -la

# ============ CELL 2: 導入庫 ============
import pandas as pd
import numpy as np
from google.colab import files
from feature_engineering import ReversalFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# ============ CELL 3: 上傳數據 ============
print("Click to upload labeled_data.csv")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

# ============ CELL 4: 載入並檢查數據 ============
df = pd.read_csv(filename)
print(f"Data loaded: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nConfirmed reversals: {(df['confirmed_label']==1).sum()}")
print(f"False signals: {(df['confirmed_label']==0).sum()}")
print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ============ CELL 5: 特徵工程 ============
# 警告: 這可能需要1-2分鐘
print("Starting feature engineering...")
print("This will compute 16 technical indicators...\n")

engine = ReversalFeatureEngineer(df)
features_df = engine.compute_all_features()

print("\nFeature engineering complete!")

# ============ CELL 6: 特徵檢查 ============
new_features = [col for col in features_df.columns if col not in df.columns]
print(f"\nTotal features created: {len(new_features)}")
print(f"\nFeature categories:")
print(f"  Momentum (4): rsi_6, rsi_14, rsi_divergence, roc_12")
print(f"  Volatility (3): bb_percent_b, bb_bandwidth, atr_14")
print(f"  Patterns (3): hammer, shooting_star, engulfing")
print(f"  Volume (3): volume_oscillator, volume_spike, volume_trend")
print(f"  Price Action (3): price_momentum, gap, higher_high_lower_low")

# ============ CELL 7: 數據樣本 ============
print("\nSample data with key features:")
display_cols = ['timestamp', 'close', 'volume', 'confirmed_label',
                'rsi_14', 'bb_percent_b', 'hammer', 'volume_spike', 'price_momentum']
print(features_df[display_cols].head(15).to_string())

# ============ CELL 8: 特徵統計 ============
confirmed = features_df[features_df['confirmed_label'] == 1]
false_sig = features_df[features_df['confirmed_label'] == 0]

print(f"\nConfirmed Reversals: {len(confirmed)} samples")
print(f"False Signals: {len(false_sig)} samples")

if len(confirmed) > 0:
    print("\n=== CONFIRMED REVERSALS - Key Feature Statistics ===")
    stats_cols = ['rsi_6', 'rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum']
    print(confirmed[stats_cols].describe().round(3))

if len(false_sig) > 0:
    print("\n=== FALSE SIGNALS - Key Feature Statistics ===")
    print(false_sig[stats_cols].describe().round(3))

# ============ CELL 9: 特徵相關性 ============
print("\nFeature Correlation with Label (confirmed_label):")
corr_with_label = features_df[new_features + ['confirmed_label']].corr()['confirmed_label'].drop('confirmed_label')
top_features = corr_with_label.abs().sort_values(ascending=False).head(10)

print("\nTop 10 most correlated features:")
for i, (feat, corr) in enumerate(top_features.items(), 1):
    actual_corr = corr_with_label[feat]
    print(f"  {i:2d}. {feat:<25} {actual_corr:>8.4f}")

# ============ CELL 10: 數據質量檢查 ============
print("\nData Quality Check:")
nan_counts = features_df[new_features].isna().sum()
print(f"  Total rows: {len(features_df)}")
print(f"  Rows with NaN: {nan_counts[nan_counts > 0].index.tolist()}")
print(f"  NaN in features: {nan_counts.sum()}")

inf_counts = np.isinf(features_df[new_features].select_dtypes(np.number)).sum().sum()
print(f"  Inf values: {inf_counts}")

# ============ CELL 11: 導出 ============
output_file = 'features_data.csv'
features_df.to_csv(output_file, index=False)
print(f"\nExported to {output_file}")
print(f"Size: {len(features_df)} rows x {len(features_df.columns)} columns")

print("\nReady to download!")

# ============ CELL 12: 下載結果 ============
files.download('features_data.csv')
print("features_data.csv downloaded!")

# 可選: 也下載統計摘要
summary = f"""PHASE 2 FEATURE ENGINEERING SUMMARY

Input: {filename}
Output: features_data.csv

Dataset Size:
  Total rows: {len(features_df)}
  Total columns: {len(features_df.columns)}
  Original columns: {len(df.columns)}
  New features: {len(new_features)}

Ground Truth:
  Confirmed Reversals: {len(confirmed)} ({len(confirmed)/len(features_df)*100:.1f}%)
  False Signals: {len(false_sig)} ({len(false_sig)/len(features_df)*100:.1f}%)

Features Created:
{new_features}

Top Correlated Features with Label:
{top_features.to_string()}
"""

with open('phase2_summary.txt', 'w') as f:
    f.write(summary)

files.download('phase2_summary.txt')
print("phase2_summary.txt downloaded!")
```

---

## 常見問題

### Q: 執行多久?
A: 通常 1-2 分鐘，取決於數據大小（10,000行約1分鐘）

### Q: 出現 "NaN" 怎麼辦?
A: 正常現象！前 35 行會有 NaN（指標初始化）。Phase 3 時會處理。

### Q: 特徵值看起來很大或很小?
A: 這是正常的！不同特徵有不同的尺度：
- RSI: 0-100
- Price Momentum: -20 to +20 (%)
- Volume Spike: 0.5 to 3.0

### Q: 可以修改參數嗎?
A: 可以！在 `feature_engineering.py` 中修改：
```python
# 例如: 改變 RSI 週期
self.df['rsi_21'] = self.compute_rsi(21)  # 改成 21 期
```

### Q: 如何在本地執行?
A: 
```bash
pip install pandas numpy
python run_phase2.py
```

---

## 下一步

導出 `features_data.csv` 後，準備進入 Phase 3：

1. **數據準備**
   ```python
   features = pd.read_csv('features_data.csv')
   features = features.iloc[35:]  # 移除 NaN
   X = features.drop(['timestamp', 'confirmed_label'], axis=1)
   y = features['confirmed_label']
   ```

2. **訓練/測試分割**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, stratify=y
   )
   ```

3. **模型訓練**
   ```python
   from lightgbm import LGBMClassifier
   model = LGBMClassifier()
   model.fit(X_train, y_train)
   ```

---

**準備好了嗎？上傳 `labeled_data.csv` 並開始執行！**

