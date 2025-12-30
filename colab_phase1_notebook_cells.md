# PHASE1 Interactive Colab Notebook Cells

这是针对Google Colab的Cell使用指南，可以在Notebook中直接调整参数。

---

## Cell 1: 安装依赖和加载数据

```python
!pip install huggingface-hub -q
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
from huggingface_hub import hf_hub_download

try:
    csv_file = hf_hub_download(
        repo_id="zongowo111/cpb-models",
        filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(csv_file)
    print(f"Successfully loaded: {df.shape}")
except Exception as e:
    print(f"Error: {e}")
    df = None

if df is not None:
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    
    if 'timestamp' not in df.columns:
        df['timestamp'] = df.get('open_time', range(len(df)))
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {str(df['timestamp'].iloc[0])[:19]} to {str(df['timestamp'].iloc[-1])[:19]}")
```

---

## Cell 2: 定义反转点检测函数

```python
def detect_reversals(df, lookback=5, confirm_local=True):
    """
    检测反转点
    """
    reversals = []
    
    def calculate_direction(idx, n):
        if idx < n:
            return 0
        return df.iloc[idx]['close'] - df.iloc[idx-n]['close']
    
    def is_local_high(idx):
        if idx <= 0 or idx >= len(df) - 1:
            return False
        current = df.iloc[idx]['close']
        prev = df.iloc[idx-1]['close']
        next_val = df.iloc[idx+1]['close']
        return current >= prev and current >= next_val
    
    def is_local_low(idx):
        if idx <= 0 or idx >= len(df) - 1:
            return False
        current = df.iloc[idx]['close']
        prev = df.iloc[idx-1]['close']
        next_val = df.iloc[idx+1]['close']
        return current <= prev and current <= next_val
    
    print(f"Detecting reversals (lookback={lookback}, confirm_local={confirm_local})...")
    
    for i in range(lookback + 1, len(df) - 1):
        prev_direction = calculate_direction(i-1, lookback)
        current_direction = calculate_direction(i, 1)
        
        # 高点反转
        if prev_direction > 0 and current_direction < 0:
            if confirm_local and not is_local_high(i-1):
                continue
            reversals.append({
                'index': i-1,
                'timestamp': df.iloc[i-1]['timestamp'],
                'close': df.iloc[i-1]['close'],
                'type': 'Local_High'
            })
        
        # 低点反转
        elif prev_direction < 0 and current_direction > 0:
            if confirm_local and not is_local_low(i-1):
                continue
            reversals.append({
                'index': i-1,
                'timestamp': df.iloc[i-1]['timestamp'],
                'close': df.iloc[i-1]['close'],
                'type': 'Local_Low'
            })
    
    return pd.DataFrame(reversals) if reversals else pd.DataFrame()

print("Function defined: detect_reversals()")
print("Ready to run detection with custom parameters")
```

---

## Cell 3: 运行检测 (参数调整在这里)

```python
# ============================================================================
# 调整参数 - 修改下面的值后重新运行此Cell
# ============================================================================

LOOKBACK = 5          # 向后看N根K线 (推荐: 1-10)
CONFIRM_LOCAL = True  # 是否验证局部高低点 (True/False)

print(f"\n{'='*70}")
print("PHASE1 Direction Change Detection - Interactive")
print(f"{'='*70}")
print(f"\n参数设置:")
print(f"  LOOKBACK: {LOOKBACK}")
print(f"  CONFIRM_LOCAL: {CONFIRM_LOCAL}")

if df is not None:
    # 检测反转点
    reversals_df = detect_reversals(df, lookback=LOOKBACK, confirm_local=CONFIRM_LOCAL)
    
    # 标记数据集
    labeled_df = df.copy()
    labeled_df['Reversal_Label'] = 0
    labeled_df['Reversal_Type'] = 'None'
    labeled_df['Reversal_Price'] = 0.0
    
    for idx, row in reversals_df.iterrows():
        i = int(row['index'])
        if i < len(labeled_df):
            labeled_df.at[i, 'Reversal_Label'] = 1
            labeled_df.at[i, 'Reversal_Type'] = row['type']
            labeled_df.at[i, 'Reversal_Price'] = row['close']
    
    # 保存
    output_file = f'phase1_lookback{LOOKBACK}_local{str(CONFIRM_LOCAL)[0]}.csv'
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
            'Reversal_Label', 'Reversal_Type', 'Reversal_Price']
    existing_cols = [c for c in cols if c in labeled_df.columns]
    labeled_df[existing_cols].to_csv(output_file, index=False)
    
    # 统计
    print(f"\n{'='*70}")
    print("PHASE1 Detection Results")
    print(f"{'='*70}")
    
    total = len(reversals_df)
    local_lows = len(reversals_df[reversals_df['type'] == 'Local_Low']) if total > 0 else 0
    local_highs = len(reversals_df[reversals_df['type'] == 'Local_High']) if total > 0 else 0
    
    print(f"\nTotal reversal points: {total}")
    print(f"  - Local Lows (Buy Points): {local_lows}")
    print(f"  - Local Highs (Sell Points): {local_highs}")
    print(f"\nReversal ratio: {total / len(df) * 100:.2f}%")
    if total > 0:
        print(f"Average spacing: {len(df) // (total + 1):.0f} bars")
    
    print(f"\nOutput file: {output_file}")
    
    if total > 0:
        print(f"\nFirst 15 reversals:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].head(15))
        
        print(f"\nLast 15 reversals:")
        print(reversals_df[['index', 'timestamp', 'type', 'close']].tail(15))
    
    print(f"\n{'='*70}")
else:
    print("ERROR: Data not loaded")
```

---

## Cell 4: 可视化反转点

```python
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = (15, 6)

if df is not None and len(reversals_df) > 0:
    # 使用最近500根K线
    start_idx = max(0, len(df) - 500)
    plot_df = labeled_df.iloc[start_idx:].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 绘制K线
    ax.plot(range(len(plot_df)), plot_df['close'], color='gray', linewidth=1, label='Close Price')
    
    # 标记Low点 (绿色)
    lows = plot_df[plot_df['Reversal_Type'] == 'Local_Low']
    ax.scatter(lows.index, lows['close'], color='green', marker='^', s=100, label='Local Low', zorder=5)
    
    # 标记High点 (红色)
    highs = plot_df[plot_df['Reversal_Type'] == 'Local_High']
    ax.scatter(highs.index, highs['close'], color='red', marker='v', s=100, label='Local High', zorder=5)
    
    ax.set_xlabel('K线索引')
    ax.set_ylabel('价格 (USDT)')
    ax.set_title(f'PHASE1 反转点检测 (lookback={LOOKBACK}, confirm_local={CONFIRM_LOCAL})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 统计最近500根中的反转点
    recent_reversals = len(plot_df[plot_df['Reversal_Label'] == 1])
    print(f"\n最近500根K线中检测到 {recent_reversals} 个反转点")
else:
    print("No reversals to visualize")
```

---

## 使用指南

### 第一次运行

1. 依次执行 Cell 1, 2, 3, 4
2. 查看输出结果和图表

### 调整参数

只需修改 **Cell 3** 中的参数：

```python
LOOKBACK = 5          # 改成 3, 4, 6, 7 等试试
CONFIRM_LOCAL = True  # 改成 False 试试
```

然后重新运行 Cell 3 即可，无需重新运行 Cell 1 和 2。

### 参数说明

| 参数 | 值 | 效果 |
|------|-----|------|
| LOOKBACK | 1 | 最灵敏，反转点最多 |
| LOOKBACK | 3 | 中等灵敏度 |
| LOOKBACK | 5 | 推荐值，平衡 |
| LOOKBACK | 7-10 | 保守，反转点最少 |
| | | |
| CONFIRM_LOCAL | True | 验证局部高低点，噪音少 |
| CONFIRM_LOCAL | False | 不验证，反转点更多 |

---

## 参数调整建议

### 如果反转点太少 (< 10%)
```python
LOOKBACK = 3          # 减小
CONFIRM_LOCAL = False # 关闭验证
```

### 如果反转点太多 (> 30%)
```python
LOOKBACK = 7          # 增大
CONFIRM_LOCAL = True  # 启用验证
```

### 要找到最佳点
```python
# 试试这个组合
LOOKBACK = 5
CONFIRM_LOCAL = True
```

---

## 导出结果

运行完 Cell 3 后，输出文件会自动保存为CSV，文件名示例：
- `phase1_lookback5_localT.csv` (LOOKBACK=5, CONFIRM_LOCAL=True)
- `phase1_lookback3_localF.csv` (LOOKBACK=3, CONFIRM_LOCAL=False)

可以在Colab左侧文件管理器中下载。
