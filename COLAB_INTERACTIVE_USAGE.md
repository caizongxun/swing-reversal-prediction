# Colab交互式参数调整使用指南

---

## 方法1: Cell变量调整 (推荐)

这是在Colab Notebook中最灵活的方法。

### 步骤1: 复制Notebook Cell

从GitHub复制 [colab_phase1_notebook_cells.md](https://github.com/caizongxun/swing-reversal-prediction/blob/main/colab_phase1_notebook_cells.md) 中的所有Cell到您的Colab Notebook。

### 步骤2: 修改参数

在 **Cell 3** 中修改参数：

```python
# ============================================================================
# 调整参数 - 修改下面的值后重新运行此Cell
# ============================================================================

LOOKBACK = 5          # 改成 1-10 之间的任意值
CONFIRM_LOCAL = True  # 改成 True 或 False
```

### 步骤3: 重新运行Cell 3

- 修改参数后直接按 `Ctrl+Enter` 或点击Run按钮
- 无需重新运行 Cell 1 和 2
- 新的结果会立即显示

### 步骤4: 查看结果和图表

- Cell 3 输出统计信息
- Cell 4 自动生成可视化图表

---

## 方法2: 命令行参数 (快速脚本)

适合快速测试不同参数组合。

### 执行命令

```bash
!python colab_phase1_direction_change_interactive.py --lookback 5 --confirm_local True
```

### 参数说明

```bash
--lookback INT        # 向后看N根K线 (default: 5)
--confirm_local BOOL  # 是否验证局部高低点 (default: True)
--output FILE         # 输出文件名 (default: labeled_klines_phase1_direction_change.csv)
```

### 示例

```python
# 例1: lookback=5, 启用验证
!python colab_phase1_direction_change_interactive.py --lookback 5 --confirm_local True

# 例2: lookback=3, 禁用验证
!python colab_phase1_direction_change_interactive.py --lookback 3 --confirm_local False

# 例3: lookback=7, 自定义输出文件
!python colab_phase1_direction_change_interactive.py --lookback 7 --output my_result.csv

# 例4: 下载脚本后执行
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_direction_change_interactive.py | python3 -- --lookback 6 --confirm_local False
```

---

## 参数调整建议

### LOOKBACK参数影响

| 值 | 结果 | 用途 |
|----|------|------|
| 1 | 3500-4000个反转点 | 捕捉所有小波动 |
| 2 | 2500-3000个反转点 | 捕捉中等波动 |
| 3 | 1800-2200个反转点 | 平衡灵敏度 |
| 5 | 800-1200个反转点 | 推荐值 |
| 7 | 400-600个反转点 | 捕捉大趋势 |
| 10 | 200-300个反转点 | 只要极端反转 |

### CONFIRM_LOCAL参数影响

```
CONFIRM_LOCAL = True
  优点: 反转点质量高，噪音少
  缺点: 数量会减少20-30%

CONFIRM_LOCAL = False
  优点: 捕捉更多可能的反转点
  缺点: 可能包含一些噪音
```

---

## 快速开始 (3步)

### 如果您已有Colab Notebook

#### 方式A: 用Cell变量 (推荐)

1. 新建一个Cell，粘贴以下代码：

```python
# 安装依赖
!pip install huggingface-hub -q
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 加载数据
from huggingface_hub import hf_hub_download
csv_file = hf_hub_download(
    repo_id="zongowo111/cpb-models",
    filename="klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv",
    repo_type="dataset"
)
df = pd.read_csv(csv_file)
df['close'] = pd.to_numeric(df['close'])
df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
if 'timestamp' not in df.columns:
    df['timestamp'] = df.get('open_time', range(len(df)))
print(f"Data loaded: {df.shape}")
```

2. 新建一个Cell，粘贴以下代码：

```python
def detect_reversals(df, lookback=5, confirm_local=True):
    reversals = []
    def calculate_direction(idx, n):
        if idx < n: return 0
        return df.iloc[idx]['close'] - df.iloc[idx-n]['close']
    def is_local_high(idx):
        if idx <= 0 or idx >= len(df) - 1: return False
        current = df.iloc[idx]['close']
        prev = df.iloc[idx-1]['close']
        next_val = df.iloc[idx+1]['close']
        return current >= prev and current >= next_val
    def is_local_low(idx):
        if idx <= 0 or idx >= len(df) - 1: return False
        current = df.iloc[idx]['close']
        prev = df.iloc[idx-1]['close']
        next_val = df.iloc[idx+1]['close']
        return current <= prev and current <= next_val
    
    for i in range(lookback + 1, len(df) - 1):
        prev_direction = calculate_direction(i-1, lookback)
        current_direction = calculate_direction(i, 1)
        if prev_direction > 0 and current_direction < 0:
            if confirm_local and not is_local_high(i-1): continue
            reversals.append({'index': i-1, 'timestamp': df.iloc[i-1]['timestamp'], 
                            'close': df.iloc[i-1]['close'], 'type': 'Local_High'})
        elif prev_direction < 0 and current_direction > 0:
            if confirm_local and not is_local_low(i-1): continue
            reversals.append({'index': i-1, 'timestamp': df.iloc[i-1]['timestamp'],
                            'close': df.iloc[i-1]['close'], 'type': 'Local_Low'})
    return pd.DataFrame(reversals) if reversals else pd.DataFrame()

print("Function ready")
```

3. 新建一个Cell，**每次修改参数后重新运行这个Cell**：

```python
# 修改这两个参数
LOOKBACK = 5
CONFIRM_LOCAL = True

# 然后运行
reversals_df = detect_reversals(df, lookback=LOOKBACK, confirm_local=CONFIRM_LOCAL)

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

total = len(reversals_df)
lows = len(reversals_df[reversals_df['type'] == 'Local_Low']) if total > 0 else 0
highs = len(reversals_df[reversals_df['type'] == 'Local_High']) if total > 0 else 0

print(f"\nResults:")
print(f"Total: {total} | Lows: {lows} | Highs: {highs}")
print(f"Ratio: {total/len(df)*100:.2f}% | Spacing: {len(df)//(total+1):.0f} bars")

if total > 0:
    print(f"\nFirst 5:")
    print(reversals_df[['index', 'timestamp', 'type', 'close']].head(5))
```

#### 方式B: 用命令行参数

1. 新建一个Cell，复制完整的 `colab_phase1_direction_change_interactive.py` 代码
2. 或直接用：

```python
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_direction_change_interactive.py | python3 -- --lookback 5 --confirm_local True
```

---

## 对比不同参数

如果您想快速对比多个参数组合：

```python
# 创建对比表格
results = []

for lookback in [3, 5, 7]:
    for confirm in [True, False]:
        rev_df = detect_reversals(df, lookback=lookback, confirm_local=confirm)
        total = len(rev_df)
        lows = len(rev_df[rev_df['type'] == 'Local_Low']) if total > 0 else 0
        results.append({
            'lookback': lookback,
            'confirm_local': confirm,
            'total': total,
            'ratio': f"{total/len(df)*100:.1f}%",
            'lows': lows,
            'highs': total - lows
        })

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))
```

---

## 常见问题

### Q: 如何在Colab中下载结果文件?
A: 运行Cell后，左侧文件管理器会显示CSV文件，点击即可下载

### Q: 参数改了但结果没变?
A: 确保重新运行了Cell 3，可能还在用缓存

### Q: 怎样知道参数是否最优?
A: 看可视化图表，确保标记的点都在局部转折处，不在平坦区域

### Q: lookback太大会怎样?
A: 反转点会很少，可能遗漏真正的交易机会

---

## 下一步

调参满意后，您可以：

1. 导出标记好的CSV文件
2. 继续进行PHASE2: 特征提取
3. PHASE3: 模型训练
4. PHASE4: 规则生成

---

## GitHub文件位置

- `colab_phase1_direction_change_interactive.py` - 命令行版本
- `colab_phase1_notebook_cells.md` - Notebook Cell模板
- `PHASE1_DIRECTION_CHANGE_GUIDE.md` - 完整说明文档
