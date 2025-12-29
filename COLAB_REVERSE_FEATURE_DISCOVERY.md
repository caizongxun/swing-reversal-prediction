# Reverse Feature Discovery - Colab Guide

本指南將指導您在Colab上進行逆向特徵提取，以發現可預測反轉的數學公式。

## 系統概述

本系統的創新在於：
1. 從**已確認的真實反轉點**出發
2. 提取其周圍的數學特徵
3. 識別區分真實信號 vs 假信號的關鍵模式
4. 自動生成AI可學習的公式候選

## 完整流程

### 第1步：在Colab中設置

```python
# Cell 1: 克隆倉庫並安裝依賴
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
!pip install pandas numpy matplotlib scikit-learn -q
```

### 第2步：準備K線數據

```python
# Cell 2: 上傳K線數據
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"上傳文件: {filename}")
```

**數據格式要求：**
- CSV格式
- 列: timestamp, open, high, low, close, volume
- 至少10,000根K棒
- 時間序列須為遞增順序

### 第3步：運行逆向特徵發現管線

```python
# Cell 3: 執行完整管線
!python colab_reverse_feature_discovery.py \
    --data_path {filename} \
    --max_rows 10000 \
    --window 5 \
    --future_candles 12 \
    --threshold 1.0 \
    --output_prefix reversal
```

### 第4步：下載結果

```python
# Cell 4: 下載所有結果文件
from google.colab import files

files.download('reversal_labeled_data_10k.csv')
files.download('reversal_extracted_features_10k.csv')
files.download('reversal_pattern_analysis_10k.txt')
files.download('reversal_formula_candidates_10k.txt')
files.download('reversal_discovery_summary_10k.txt')
```

## 五個分析階段詳解

### Phase 1: 反轉點檢測

輸入：10,000根K線數據

過程：
- 使用局部最高/最低值識別原始振幅點
- 檢查未來12根K棒是否有≥1%的價格移動
- 確認真實反轉點

輸出示例：
```
Raw Swing Points: 1,250
Confirmed Reversals: 135
True Reversal Rate: 10.8%
```

### Phase 2: 數學特徵提取

從每個反轉點周圍提取以下特徵（向後看20根K棒）：

**價格特徵：**
- close_change: 價格變化百分比
- high_range: 高-低範圍
- volatility: 波動率

**交易量特徵：**
- volume_sma_ratio: 當前交易量/平均交易量
- volume_trend: 交易量變化

**技術指標：**
- RSI: 相對強度指數
- MACD: 移動平均收斂散度
- Bollinger Bands: 布林帶位置和擠壓度
- ATR: 平均真實範圍

**總計：18個數學特徵**

### Phase 3: 模式識別

比較確認反轉 vs 假信號的特徵：

```
Top Features by Difference Ratio:
1. volume_sma_ratio: Confirmed=1.45, False=0.92, Ratio=0.58
2. rsi: Confirmed=28.3, False=45.2, Ratio=0.37
3. bb_position: Confirmed=0.15, False=0.52, Ratio=0.71
...
```

### Phase 4: 公式生成

AI自動生成候選公式，包括：

**數值公式（用於評分）：**
```
(volume_sma_ratio + volatility) / 2
(RSI - 50) * volume_trend
bb_position * (1 - bb_squeeze)
```

**邏輯條件（用於分類）：**
```
RSI < 30 AND volume_sma_ratio > 1.2
MACD > macd_signal AND volume_trend > 0
bb_position < 0.2 AND atr_ratio > 0.01
```

### Phase 5: 結果導出

生成5個輸出文件：

1. **reversal_labeled_data_10k.csv**
   - 標記了所有反轉點的K線數據
   - 用於後續模型訓練

2. **reversal_extracted_features_10k.csv**
   - 確認反轉和假信號的特徵集合
   - 用於特徵分析

3. **reversal_pattern_analysis_10k.txt**
   - 20個最重要的特徵差異分析
   - 說明每個特徵的區分度

4. **reversal_formula_candidates_10k.txt**
   - AI生成的所有候選公式
   - 數值公式和邏輯條件

5. **reversal_discovery_summary_10k.txt**
   - 執行摘要和統計數據

## 下一步：AI模型訓練

使用生成的文件進行：

1. **特徵工程**
   - 測試候選公式的有效性
   - 計算特徵重要性
   - 優化特徵組合

2. **模型訓練**
   - 訓練分類器預測反轉
   - 驗證模型性能
   - 參數優化

3. **實時應用**
   - 在新K線上應用公式
   - 實時預測反轉
   - 評估交易信號

## 參數說明

| 參數 | 默認值 | 說明 |
|------|-------|------|
| data_path | - | K線CSV文件路徑（必須） |
| max_rows | 10000 | 最多分析多少根K棒 |
| window | 5 | 反轉檢測窗口大小 |
| future_candles | 12 | 檢查未來多少根K棒確認反轉 |
| threshold | 1.0 | 價格移動確認閾值（%） |
| output_prefix | reversal | 輸出文件前綴 |

## 故障排除

### 導入錯誤
```
ModuleNotFoundError: No module named 'swing_reversal_detector'
```
**解決方案：** 確保所有Python文件在同一目錄中

### 數據格式錯誤
```
KeyError: 'open' (or other column)
```
**解決方案：** 檢查CSV列名是否為: timestamp, open, high, low, close, volume

### 內存不足
**解決方案：** 降低max_rows參數，例如：`--max_rows 5000`

## 技術詳節

### 特徵提取算法

對於每個反轉點，系統：
1. 提取向後20根K棒的窗口
2. 計算所有18個數學特徵
3. 標記為確認反轉（1）或假信號（0）
4. 保存特徵向量

### 模式識別方法

使用統計差異比來識別關鍵特徵：
```
Difference Ratio = |confirmed_mean - false_mean| / |false_mean|
```

排序後的特徵表示AI應該關注哪些。

### 公式生成策略

1. **單特徵公式：** 標準化每個關鍵特徵
2. **組合公式：** 特徵的算術組合（+, -, *, /）
3. **邏輯公式：** 閾值和AND條件

## 預期結果

在10,000根BTCUSDT 15分鐘K線上：

- **確認反轉數量：** ~100-150個
- **真實反轉率：** ~10-15%
- **提取的特徵：** 18個數學指標
- **識別的模式：** ~20-30個高差異特徵
- **生成的公式：** ~50-100個候選

## 相關文件

- `phase1_analysis_10k.py` - 反轉檢測引擎（10k優化）
- `swing_reversal_detector.py` - 核心檢測算法
- `reverse_feature_extraction.py` - 特徵提取模塊
- `colab_reverse_feature_discovery.py` - 完整管線

## 參考

項目倉庫：https://github.com/caizongxun/swing-reversal-prediction

## 聯繫

如有問題，請在GitHub倉庫提交Issue。
