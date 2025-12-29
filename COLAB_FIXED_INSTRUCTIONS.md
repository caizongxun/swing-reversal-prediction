# Colab 修正後執行指南

## 所有檔案都已修正，編碼錯誤已解決

已修正的檔案：
- `colab_fetch_from_huggingface.py` - 獨立數據下載腳本
- `colab_hf_full_pipeline.py` - 完整管道腳本（下載+分析）

---

## 在 Colab 中執行

### Step 1：重啟 Colab 運行時（如果之前有錯誤）

運行時 → 重啟運行時

### Step 2：在新的 Cell 中克隆倉庫

```python
!rm -rf /content/swing-reversal-prediction 2>/dev/null || true
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction

# 驗證檔案
!ls *.py | head
```

### Step 3：執行修正後的腳本

**選項 A：只下載數據**
```python
!python colab_fetch_from_huggingface.py
```

**選項 B：下載 + 分析（推薦）**
```python
!python colab_hf_full_pipeline.py
```

---

## 預期輸出

### 成功執行的標誌

```
安裝必需的庫...
依賴庫安裝完成

可用的數據列表:
交易對列表:
  - BTCUSDT
  - ETHUSDT
  ...

支持的時間週期:
  - 15m
  - 1h

數據來源: https://huggingface.co/datasets/zongowo111/cpb-models/tree/main/klines_binance_us

開始下載K線數據...
下載 BTCUSDT 15m...
成功: 10000 行數據

數據已保存，可以用於後續分析
```

### 分析腳本的輸出

```
============================================================
# HuggingFace K線數據 + 反轉檢測完整管道
============================================================

步驟 1: 從HuggingFace下載K線數據
------------------------------------------------------------

安裝依賴庫...
依賴庫安裝完成

是否互動式選擇數據? (y/n, 默認 n): 

使用默認數據: BTCUSDT 15分鐘
下載 BTCUSDT 15m...
成功: 10000 行數據

下載完成: 10000 行數據
日期範圍: 2025-XX-XX 至 2025-XX-XX

步驟 2: 執行反轉檢測分析
------------------------------------------------------------

============================================================
開始反轉檢測分析
============================================================
Phase 1: 檢測振幅點...
  檢測到 XXX 個局部最高點
  檢測到 XXX 個局部最低點

Phase 2: 驗證反轉...
  確認 XXX 個反轉點

Phase 3: 提取特徵...
  提取 XXX 組特徵

Phase 4: 導出結果...
  已保存特徵檔案: hf_klines_analysis_result.csv
  已保存標記數據: hf_klines_labeled_data.csv

分析完成！
============================================================

分析統計:
  確認反轉點: XXX
  提取特徵數: XXX

所有結果檔案已生成
```

---

## 如果仍有問題

### 問題 1：還是沒有輸出

在 Colab 中逐行診斷：

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'colab_fetch_from_huggingface.py'],
    capture_output=True,
    text=True,
    timeout=120
)

print("輸出:")
print(result.stdout)
print("\n錯誤:")
print(result.stderr)
print(f"\n返回碼: {result.returncode}")
```

### 問題 2：找不到檔案

```python
import os
print("當前目錄:", os.getcwd())
print("\n檔案列表:")
os.system('ls -la *.py')
```

### 問題 3：依賴包錯誤

```python
!pip install -q datasets huggingface-hub pandas numpy
```

---

## 生成的輸出檔案

執行完成後，Colab 會生成：

1. **hf_klines_labeled_data.csv** - K線數據 + 反轉標籤
   - 列: timestamp, open, high, low, close, volume, reversal_label

2. **hf_klines_analysis_result.csv** - 特徵數據
   - 列: index, type, move_pct, volume_ratio, price_change, volatility

### 下載檔案到本地

```python
from google.colab import files

files.download('hf_klines_labeled_data.csv')
files.download('hf_klines_analysis_result.csv')
```

---

## 重要提示

✓ 所有 `\n` 編碼錯誤已修正
✓ Python 代碼現在能正常執行
✓ 檔案已推送到 GitHub 主分支
✓ 重新克隆後可立即使用
