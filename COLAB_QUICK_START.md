# Colab 快速開始指南

## 整體流程

你現在可以在 Google Colab 中完整執行整個 Phase 2 + Phase 2.5 + Phase 3 的流程。

```
Colab 中執行 → 自動從 GitHub 拉代碼 → 從 HuggingFace 下載 BTC 15m 數據
     ↓
 Phase 2: 計算 16 個技術指標
     ↓
 Phase 2.5: 提取反轉樣本
     ↓
 Phase 3: 特徵探索 & 發現公式
     ↓
 下載結果
```

---

## 在 Colab 中執行（5 分鐘）

### 方式 1: 直接執行遠端腳本（推薦）

在 Google Colab 中新建筆記本，執行以下代碼：

```python
# Cell 1: 執行遠端腳本
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction

!python run_phase2_phase2_5_colab.py
```

就這樣！系統會自動：
1. 克隆倉庫
2. 安裝依賴
3. 從 HuggingFace 下載 BTC 15m 數據
4. 執行 Phase 2 + Phase 2.5
5. 生成 3 個 CSV 文件
6. 自動下載到本地

### 方式 2: 分步執行（便於調試）

如果你想分步執行和檢查，可以分別運行：

```python
# Cell 1: Clone & Install
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
!pip install huggingface-hub tqdm -q
```

```python
# Cell 2: Phase 2 + Phase 2.5
!python run_phase2_phase2_5_colab.py
```

```python
# Cell 3: Phase 3 (可選)
!python feature_exploration.py
```

---

## 輸出文件

執行完成後，你會得到：

### Phase 2 輸出
```
BTCUSDT_15m_features.csv
├─ 10,000 行
├─ 27 列 (11 個 OHLCV + 16 個技術指標)
└─ 大小: ~5-10 MB
```

### Phase 2.5 輸出
```
BTCUSDT_15m_samples.csv
├─ 反轉樣本 + 非反轉樣本
├─ 每個樣本包含 lookback 的特徵統計
├─ 標籤: is_reversal (0 或 1)
└─ 用於訓練模型

BTCUSDT_15m_feature_stats.csv
└─ 特徵統計信息 (mean, std, min, max)
```

### Phase 3 輸出（可選）
```
BTCUSDT_15m_feature_importance.csv
├─ Random Forest 計算的特徵重要性
└─ 排序：最重要 → 最不重要

BTCUSDT_15m_tree_rules.txt
├─ 決策樹提取的人類可讀規則
├─ 例: IF rsi_14 < 30 AND volume_spike > 1.5 THEN reversal
└─ 這就是你的「反轉公式」

BTCUSDT_15m_feature_interactions.csv
└─ 特徵之間的相互作用分析
```

---

## 輸出示例

### 特徵重要性 (feature_importance.csv)
```
feature                    importance
rsi_14_current             0.2341
bb_percent_b_current       0.1892
volume_spike_current       0.1456
rsi_divergence_current     0.1203
price_momentum_current     0.0987
...
```

### 決策樹規則 (tree_rules.txt)
```
|--- rsi_14_current <= 35.00
|   |--- bb_percent_b_current <= 0.45
|   |   |--- volume_spike_current <= 1.20
|   |   |   |--- class: 1 (Reversal)
|   |   |--- volume_spike_current > 1.20
|   |   |   |--- class: 0 (Non-Reversal)
|   |--- bb_percent_b_current > 0.45
|   |   |--- class: 0
|--- rsi_14_current > 35.00
|   |--- class: 0
```

### 特徵互作用 (feature_interactions.csv)
```
feature              reversal_mean  non_reversal_mean  mean_ratio
rsi_14_current       25.3          65.2               0.388
bb_percent_b_current 0.15          0.52               0.288
volume_spike_current 1.85          1.12               1.652
```

---

## 常見問題

### Q1: 執行多久？

| 步驟 | 耗時 |
|------|------|
| Clone + Install | 30 秒 |
| 下載 BTC 15m 數據 | 2-3 分鐘 |
| Phase 2 (計算特徵) | 2-3 分鐘 |
| Phase 2.5 (提取樣本) | 1 分鐘 |
| Phase 3 (模型訓練) | 2-3 分鐘 |
| **總計** | **~10 分鐘** |

### Q2: 下載失敗怎麼辦？

**A**: 可能是 HuggingFace 連接超時。可以手動重試：

```python
# 在 Colab 中手動下載
from huggingface_hub import hf_hub_download

csv_path = hf_hub_download(
    repo_id="zongowo111/cpb-models",
    filename="BTCUSDT/BTCUSDT_15m_binance_us.csv",
    repo_type="dataset",
    force_download=True,  # 強制重新下載
    timeout=180  # 增加超時時間
)
```

### Q3: 我想改成其他幣種或時間框架怎麼辦？

**A**: 編輯 `run_phase2_phase2_5_colab.py` 的開頭：

```python
PAIR = "ETHUSDT"  # 改成 ETH
TIMEFRAME = "1h"   # 改成 1h
```

然後重新執行。

### Q4: 輸出文件太多怎麼辦？

**A**: Colab 會自動打包下載。如果想手動管理，可以：

```python
from google.colab import files

# 只下載特定文件
files.download('BTCUSDT_15m_samples.csv')
files.download('BTCUSDT_15m_feature_importance.csv')
```

---

## 下一步：批量處理所有幣種

如果要處理所有 17 個幣種 × 2 個時間框架 = 34 個配置：

```python
# 在 Colab 中
import subprocess

PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", ...]
TIMEFRAMES = ["15m", "1h"]

for pair in PAIRS:
    for timeframe in TIMEFRAMES:
        print(f"\n處理 {pair} {timeframe}...")
        
        # 修改配置
        subprocess.run([
            "python", "run_phase2_phase2_5_colab.py",
            "--pair", pair,
            "--timeframe", timeframe
        ])
```

**但建議先在 BTC 15m 上驗證，然後再擴展到其他配置。**

---

## 文件速查表

| 文件 | 目的 | 用途 |
|------|------|------|
| `feature_engineering.py` | Phase 2 模組 | 計算 16 個技術指標 |
| `feature_sampling.py` | Phase 2.5 模組 | 從反轉點提取樣本 |
| `feature_exploration.py` | Phase 3 模組 | 特徵探索 & 公式發現 |
| `run_phase2_phase2_5_colab.py` | **Colab 執行腳本** | 在 Colab 一鍵執行全流程 |
| `COLAB_PHASE2_WITH_HF_DATA.md` | 詳細文檔 | 手動分步執行指南 |
| `COLAB_QUICK_START.md` | 本文件 | 快速開始 |

---

## 架構圖

```
https://huggingface.co/datasets/zongowo111/cpb-models
                         ↓
          BTCUSDT_15m_binance_us.csv (10,000 rows)
                         ↓
            [Phase 2: Feature Engineering]
                    ↓ feature_engineering.py
              compute_all_features()
                    ↓
          BTCUSDT_15m_features.csv (27 columns)
                    ↓ OHLCV + 16 指標
            [Phase 2.5: Feature Sampling]
                    ↓ feature_sampling.py
              extract_reversal_samples()
                    ↓
          BTCUSDT_15m_samples.csv (平衡數據集)
                    ↓ 反轉 + 非反轉樣本
            [Phase 3: Feature Exploration]
                    ↓ feature_exploration.py
              Random Forest + Decision Tree
                    ↓
        ┌─────────┬──────────┬──────────┐
        ↓         ↓          ↓          ↓
    Importance  TreeRules  Interactions  Summary
        
    *** 這些就是你的「反轉公式」***
```

---

## 立即開始

1. **打開 Google Colab**: https://colab.research.google.com
2. **新建筆記本**
3. **執行第一個 Cell**:

```python
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
!python run_phase2_phase2_5_colab.py
```

4. **等待完成**（約 10 分鐘）
5. **下載結果**（自動彈窗）

**就這樣！** 你現在擁有了 BTC 15m 的完整反轉預測特徵數據集。

---

## 支援

遇到問題？
- 檢查 GitHub Issues
- 查看詳細文檔: `COLAB_PHASE2_WITH_HF_DATA.md`
- 確保網路連接正常（HuggingFace 可能有地區限制）

**祝你交易成功！** 🚀
