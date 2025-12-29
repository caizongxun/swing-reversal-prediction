# Phase 2 在 Google Colab 執行 - 使用 HuggingFace 數據集

## 數據集結構

你的 HuggingFace 數據集位於：
```
https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/klines_binance_us
```

### 文件結構
```
klines_binance_us/
├── BTCUSDT/
│   ├── BTCUSDT_15m_binance_us.csv (10,000 rows)
│   └── BTCUSDT_1h_binance_us.csv (10,000 rows)
├── ETHUSDT/
│   ├── ETHUSDT_15m_binance_us.csv
│   └── ETHUSDT_1h_binance_us.csv
├── ... (17 交易對)
└── APEUSDT/
    ├── APEUSDT_15m_binance_us.csv
    └── APEUSDT_1h_binance_us.csv
```

**共計**：17 個加密貨幣對 × 2 個時間框架 = 34 個 CSV 文件，每個 10,000 行

---

## Colab 快速開始（10 分鐘）

### 步驟 1：克隆倉庫和安裝依賴

```python
# Cell 1: Clone and Install
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction

# Install dependencies
!pip install pandas numpy huggingface-hub tqdm -q

print("Setup complete!")
```

### 步驟 2：從 HuggingFace 下載單個交易對數據

```python
# Cell 2: Download Data from HuggingFace
from huggingface_hub import hf_hub_download
import pandas as pd
import os

# 配置
REPO_ID = "zongowo111/cpb-models"
PAIR = "BTCUSDT"  # 可改成其他交易對
TIMEFRAME = "15m"  # 或 "1h"

# 文件名
filename = f"{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv"

print(f"正在從 HuggingFace 下載: {PAIR} {TIMEFRAME}...")

# 下載
csv_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=filename,
    repo_type="dataset",
    cache_dir="./data"
)

print(f"下載完成: {csv_path}")

# 載入數據
df = pd.read_csv(csv_path)
print(f"\n數據大小: {df.shape}")
print(f"\n列名:")
print(df.columns.tolist())
print(f"\n前 5 行:")
print(df.head())
```

### 步驟 3：導入特徵工程模組

```python
# Cell 3: Import Feature Engineering Module
import sys
sys.path.insert(0, '/content/swing-reversal-prediction')

from feature_engineering import ReversalFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

print("特徵工程模組載入成功!")
```

### 步驟 4：數據預處理（轉換為標記格式）

```python
# Cell 4: Prepare Data
# 你的 CSV 需要有: timestamp, open, high, low, close, volume
# 我們需要轉換為標記格式

df_for_labeling = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

# 添加標記列（初始化為 0，Phase 1 應該已經做過）
if 'confirmed_label' not in df_for_labeling.columns:
    df_for_labeling['confirmed_label'] = 0  # 臨時設定，實際應使用 Phase 1 標籤
    df_for_labeling['swing_type'] = ''
    df_for_labeling['raw_label'] = 0
    df_for_labeling['future_move_pct'] = 0.0
    df_for_labeling['is_confirmed_reversal'] = False

print(f"準備好的數據:")
print(f"大小: {df_for_labeling.shape}")
print(f"列: {df_for_labeling.columns.tolist()}")
print(f"\n前 5 行:")
print(df_for_labeling.head())
```

### 步驟 5：執行特徵工程

```python
# Cell 5: Run Feature Engineering
print(f"開始計算特徵 ({PAIR} {TIMEFRAME})...")
print("="*60)

engineering = ReversalFeatureEngineer(df_for_labeling)
features_df = engineering.compute_all_features()

print("="*60)
print(f"特徵工程完成!")
print(f"\n結果:")
print(f"  行數: {len(features_df)}")
print(f"  列數: {len(features_df.columns)}")
print(f"  新特徵數: 16")
```

### 步驟 6：檢查特徵質量

```python
# Cell 6: Feature Quality Check
print("特徵統計信息:")
print("="*60)

feature_cols = ['rsi_6', 'rsi_14', 'bb_percent_b', 'volume_spike', 
                 'price_momentum', 'atr_14', 'hammer', 'shooting_star']

print(f"\n樣本特徵值:")
print(features_df[['timestamp'] + feature_cols].head(10))

print(f"\n特徵範圍:")
for col in feature_cols:
    print(f"  {col:20} min={features_df[col].min():8.3f}, max={features_df[col].max():8.3f}")

print(f"\nNaN 檢查:")
nan_counts = features_df[feature_cols].isna().sum()
print(f"  包含 NaN 的特徵: {nan_counts[nan_counts > 0].index.tolist()}")
if nan_counts.sum() > 0:
    print(f"  (首 35 行有 NaN 是正常的 - 指標初始化期)")
```

### 步驟 7：導出結果

```python
# Cell 7: Export Results
output_file = f"{PAIR}_{TIMEFRAME}_features.csv"
features_df.to_csv(output_file, index=False)

print(f"✓ 特徵數據已保存: {output_file}")
print(f"  大小: {len(features_df)} 行 × {len(features_df.columns)} 列")

# 下載到本地
from google.colab import files
files.download(output_file)
print(f"✓ 文件已準備下載")
```

---

## 批量處理多個交易對（進階）

### 一次處理所有交易對

```python
# Cell 8: Batch Process All Pairs
import os
from tqdm import tqdm
from huggingface_hub import hf_hub_download

REPO_ID = "zongowo111/cpb-models"
TIMEFRAME = "15m"  # 改成 "1h" 處理 1h 數據

# 所有交易對
ALL_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT",
    "UNIUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT",
    "VETUSDT", "FILUSDT", "THETAUSDT", "NEARUSDT", "APEUSDT"
]

results = {}

for pair in tqdm(ALL_PAIRS, desc=f"處理 {TIMEFRAME} 數據"):
    try:
        # 下載
        filename = f"{pair}/{pair}_{TIMEFRAME}_binance_us.csv"
        csv_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir="./data"
        )
        
        # 載入
        df = pd.read_csv(csv_path)
        
        # 準備
        df_prep = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        if 'confirmed_label' not in df_prep.columns:
            df_prep['confirmed_label'] = 0
            df_prep['swing_type'] = ''
            df_prep['raw_label'] = 0
            df_prep['future_move_pct'] = 0.0
            df_prep['is_confirmed_reversal'] = False
        
        # 計算特徵
        engineering = ReversalFeatureEngineer(df_prep)
        features_df = engineering.compute_all_features()
        
        # 保存
        output_file = f"{pair}_{TIMEFRAME}_features.csv"
        features_df.to_csv(output_file, index=False)
        
        results[pair] = {
            'status': 'success',
            'rows': len(features_df),
            'file': output_file
        }
        
    except Exception as e:
        results[pair] = {
            'status': 'error',
            'error': str(e)
        }

print("\n批量處理完成!")
print("="*60)
for pair, result in results.items():
    if result['status'] == 'success':
        print(f"✓ {pair:12} - {result['rows']} 行 → {result['file']}")
    else:
        print(f"✗ {pair:12} - 錯誤: {result['error']}")
```

### 下載所有結果

```python
# Cell 9: Download All Results
import shutil

# 建立壓縮文件
shutil.make_archive(
    'phase2_features_all',
    'zip',
    '.',
    '*.csv'  # 所有 CSV 文件
)

files.download('phase2_features_all.zip')
print("✓ 所有特徵文件已打包下載: phase2_features_all.zip")
```

---

## 合並所有交易對數據（可選）

```python
# Cell 10: Combine All Pairs
import glob

all_dataframes = []

for csv_file in sorted(glob.glob(f"*_{TIMEFRAME}_features.csv")):
    df = pd.read_csv(csv_file)
    all_dataframes.append(df)
    print(f"載入: {csv_file} ({len(df)} 行)")

# 合併
combined_df = pd.concat(all_dataframes, ignore_index=True)
print(f"\n合併結果: {len(combined_df)} 行 × {len(combined_df.columns)} 列")

# 保存
combined_df.to_csv(f'all_pairs_{TIMEFRAME}_features.csv', index=False)
print(f"✓ 已保存: all_pairs_{TIMEFRAME}_features.csv")

files.download(f'all_pairs_{TIMEFRAME}_features.csv')
```

---

## 常見問題

### Q1: 下載超時怎麼辦？

**A**: HuggingFace 有時會慢，可以重試或分多次下載：

```python
from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=filename,
    repo_type="dataset",
    cache_dir="./data",
    force_download=True,  # 強制重新下載
    timeout=120  # 增加超時時間
)
```

### Q2: 我的數據 CSV 列名不同怎麼辦？

**A**: 調整列名映射：

```python
df = pd.read_csv(csv_path)

# 如果列名不同，重命名
df = df.rename(columns={
    'your_timestamp_col': 'timestamp',
    'your_open_col': 'open',
    'your_high_col': 'high',
    'your_low_col': 'low',
    'your_close_col': 'close',
    'your_volume_col': 'volume'
})
```

### Q3: 特徵計算很慢怎麼辦？

**A**: 先在小數據集上測試：

```python
# 只用前 1000 行
df_small = df.head(1000)
engineering = ReversalFeatureEngineer(df_small)
features_df = engineering.compute_all_features()

print(f"小數據集測試: {len(features_df)} 行，耗時約 30 秒")
```

### Q4: Colab 中斷了怎麼辦？

**A**: 保存進度到 Google Drive：

```python
from google.colab import drive

drive.mount('/content/drive')

# 保存到 Google Drive
features_df.to_csv('/content/drive/My Drive/features.csv', index=False)
print("已保存到 Google Drive")
```

---

## Phase 2 完整工作流程

```
┌─────────────────────────────────┐
│ HuggingFace 數據集              │
│ (klines_binance_us)             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Cell 1-2: 下載數據              │
│ (BTCUSDT_15m_binance_us.csv)   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Cell 3-4: 準備數據              │
│ (轉換格式，添加標籤列)          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Cell 5: 特徵工程                │
│ (16 個技術指標)                 │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Cell 6-7: 檢查並導出            │
│ (BTCUSDT_15m_features.csv)      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ Cell 8-9: 批量處理所有交易對    │
│ (34 個文件)                     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 輸出: features_data.csv         │
│ (準備進入 Phase 3 模型訓練)     │
└─────────────────────────────────┘
```

---

## 預期結果

執行完成後，你會得到：

```
✓ BTCUSDT_15m_features.csv
  - 10,000 行
  - 27 列 (11 個 OHLCV + 16 個特徵)
  - 大小約 5-10 MB

✓ ETHUSDT_15m_features.csv
✓ BNBUSDT_15m_features.csv
... (34 個文件)

✓ all_pairs_15m_features.csv
  - 340,000 行 (17 pairs × 2 timeframes × 10,000 rows)
  - 準備進入 Phase 3 模型訓練
```

---

## 下一步

1. **在 Colab 中執行上述代碼**
2. **下載 `all_pairs_15m_features.csv`**
3. **進入 Phase 3: 模型訓練**

---

**準備好開始? 複製上面的代碼到 Colab 並執行吧!**

