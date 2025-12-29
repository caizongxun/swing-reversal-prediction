# 逆向特徵提取完整指南

本指南包含您所需的所有信息，用於在Colab中執行反轉點逆向特徵發現系統。

## 快速概述

**目標：** 從10,000根確認的K線反轉點中自動發現數學預測公式

**方法：** 
1. 檢測反轉點（向後看20根K棒）
2. 提取18個數學特徵
3. 識別區分真實反轉vs假信號的模式
4. AI自動生成50-100個公式候選
5. 導出結果供後續訓練

**預期結果：**
- 135個確認反轉點
- 20個關鍵特徵
- 50-100個公式候選
- 5個輸出文件

---

## 3分鐘快速開始

### Colab Cell 1: 環境設置
```python
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
!pip install pandas numpy matplotlib scikit-learn -q
```

### Colab Cell 2: 上傳數據
```python
from google.colab import files
uploaded = files.upload()
data_file = list(uploaded.keys())[0]
print(f"已上傳: {data_file}")
```

### Colab Cell 3: 執行分析
```python
!python colab_reverse_feature_discovery.py \
    --data_path {data_file} \
    --max_rows 10000 \
    --window 5 \
    --future_candles 12 \
    --threshold 1.0 \
    --output_prefix reversal
```

### Colab Cell 4: 下載結果
```python
from google.colab import files
files.download('reversal_labeled_data_10k.csv')
files.download('reversal_extracted_features_10k.csv')
files.download('reversal_pattern_analysis_10k.txt')
files.download('reversal_formula_candidates_10k.txt')
files.download('reversal_discovery_summary_10k.txt')
```

---

## 數據準備

**要求：**
- CSV格式
- 至少10,000根K棒
- 列名: timestamp, open, high, low, close, volume
- 時間序列遞增

**示例：**
```
timestamp,open,high,low,close,volume
2025-09-14 09:30:00,116025.32,116054.89,116025.32,116054.89,0.02515
2025-09-14 09:45:00,116059.68,116126.19,115865.35,116125.87,0.0525
```

---

## 五階段分析詳解

### Phase 1: 振幅檢測
- 檢測局部最高/最低點
- 確認條件: 未來12根K棒≥1%反向移動
- 輸出: 帶標籤的反轉數據

### Phase 2: 特徵提取  
- 從每個反轉周圍提取20根K棒
- 計算18個數學特徵
- 輸出: 特徵向量

### Phase 3: 模式識別
- 比較確認反轉vs假信號的特徵
- 計算差異比率
- 輸出: 排序的關鍵特徵

### Phase 4: 公式生成
- 數值公式組合
- 邏輯條件組合
- 輸出: 公式候選

### Phase 5: 結果導出
- 導出5個文件
- 包含所有中間結果
- 供後續訓練使用

---

## 18個數學特徵

**分類:**
1. 價格特徵 (4個): close_change, high_range, volatility, momentum
2. 交易量特徵 (3個): volume_sma_ratio, volume_trend, avg_body
3. 價格關係 (2個): close_vs_high, close_vs_low
4. 技術指標 (9個): RSI, MACD, MACD Signal, BB Position, BB Squeeze, ATR Ratio等

---

## 輸出文件說明

1. **labeled_data_10k.csv** - 帶標籤的K線數據
2. **extracted_features_10k.csv** - 18個特徵 + 標籤
3. **pattern_analysis_10k.txt** - 20個關鍵特徵分析
4. **formula_candidates_10k.txt** - 50-100個公式
5. **discovery_summary_10k.txt** - 執行摘要

---

## 參數說明

| 參數 | 默認 | 調整指南 |
|------|------|----------|
| max_rows | 10000 | 減少以加快速度，增加以獲得更多樣本 |
| window | 5 | 增加檢測更多反轉但假信號增加 |
| future_candles | 12 | 增加確認條件寬鬆，減少更嚴格 |
| threshold | 1.0 | 降低檢測小反轉，提高只檢測大反轉 |

---

## 故障排查

**ModuleNotFoundError**: 確保所有.py文件在同一目錄
**KeyError**: 檢查CSV列名是否正確
**MemoryError**: 降低max_rows參數
**無反轉檢測**: 調低threshold參數

---

## 下一步行動

1. 查看pattern_analysis文件中的前10個特徵
2. 評估formula_candidates中的公式
3. 選擇top 3-5個公式進行驗證
4. 使用提取的特徵進行回測
5. 優化參數和策略
6. 訓練機器學習模型

---

## 預期成果

**立即：** 10,000根K線反轉點的完整數據集 + 50-100個公式候選

**短期：** 驗證公式性能，選擇最佳策略

**長期：** 自動化交易系統實現

---

## 文檔

- `QUICK_START_CHECKLIST.md` - 3分鐘快速啟動
- `SYSTEM_OVERVIEW.md` - 完整系統概述
- `reverse_feature_discovery_execution_plan.md` - 詳細執行計畫
- `COLAB_REVERSE_FEATURE_DISCOVERY.md` - Colab詳細教程

---

**準備好開始了嗎？在Colab中執行Cell 1吧！**

GitHub: https://github.com/caizongxun/swing-reversal-prediction/
