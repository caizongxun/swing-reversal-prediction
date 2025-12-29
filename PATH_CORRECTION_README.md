# HuggingFace 数据集路径修正

## 问题

之前的脚本使用了错误的路径来访问 HuggingFace 数据集:

**错误路径：**
```
klines_binance_us/BTCUSDT_15m.csv
```

**实际路径：**
```
klines/BTCUSDT/BTCUSDT_15m_binance_us.csv
```

是什么帮疤成了正确路径：你提供的 `klines_summary_binance_us.json` 文件

### JSON 中的元数据

```json
{
  "BTCUSDT": {
    "15m": {
      "rows": 10000,
      "csv_path": "klines/BTCUSDT/BTCUSDT_15m_binance_us.csv",
      "start_time": "2025-09-14T09:30:00",
      "end_time": "2025-12-27T13:15:00"
    },
    "1h": {
      "rows": 10000,
      "csv_path": "klines/BTCUSDT/BTCUSDT_1h_binance_us.csv",
      "start_time": "2024-11-05T22:00:00",
      "end_time": "2025-12-27T13:00:00"
    }
  },
  ...
}
```

---

## 解决方案

### 新脚本：colab_hf_corrected.py

已经修正了路径访问逻辑:

```python
# 正确的路径结构
file_name = f"{trading_pair}_{timeframe}_binance_us.csv"
file_path_in_repo = f"klines/{trading_pair}/{file_name}"

# 例项：
# BTCUSDT 15m -> klines/BTCUSDT/BTCUSDT_15m_binance_us.csv
# ETHUSDT 1h  -> klines/ETHUSDT/ETHUSDT_1h_binance_us.csv
```

### 使用新脚本

在 Colab 中执行：

```python
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction

# 执行修正版脚本
!python colab_hf_corrected.py
```

---

## HuggingFace 数据集的实际结构

```
zongowo111/cpb-models
├─ klines/
│  ├─ BTCUSDT/
│  │  ├─ BTCUSDT_15m_binance_us.csv
│  │  ├─ BTCUSDT_1h_binance_us.csv
│  │  └─ ...
│  ├─ ETHUSDT/
│  │  ├─ ETHUSDT_15m_binance_us.csv
│  │  ├─ ETHUSDT_1h_binance_us.csv
│  │  └─ ...
│  ├─ BNBUSDT/
│  └─ ...
└─ models_v6/
   ├─ ADA_15m_metrics.json
   ├─ ADA_1h_metrics.json
   └─ ...
```

---

## 支持的交易对

从 JSON 文件中提取的所有交易对（您的 klines_summary_binance_us.json）：

1. BTCUSDT
2. ETHUSDT
3. BNBUSDT
4. SOLUSDT
5. XRPUSDT
6. ADAUSDT
7. DOGEUSDT
8. MATICUSDT
9. AVAXUSDT
10. LINKUSDT
11. LTCUSDT
12. UNIUSDT
13. BCHUSDT
14. ETCUSDT
15. XLMUSDT
16. VETUSDT
17. FILUSDT
18. THETAUSDT
19. NEARUSDT
20. APEUSDT

### 两个时间周期
- 15m (帕 15 分钟)
- 1h (每小时)

### 数据特点
- 每个交易对 10,000 行数据
- 时间范围: 最近 3-12 个月（取决于交易对）
- 列：timestamp, open, high, low, close, volume

---

## 集成到 trainer 项目

### 第一步：下載数据

```python
# 使用修正版脚本
from colab_hf_corrected import HuggingFaceKlinesDownloader

downloader = HuggingFaceKlinesDownloader()
df = downloader.fetch_and_combine(
    trading_pair="BTCUSDT",
    timeframe="15m",
    rows=10000
)
```

### 第二步：用于 trainer 项目

```python
# 将 df 传递给 trainer 预处理
from trainer import preprocess_klines

processed_data = preprocess_klines(df)
```

---

## Colab 中的部署流程

### 步骤 1：下載数据

```bash
!python colab_hf_corrected.py
```

输出示例（成功的情况）：

```
============================================================
HuggingFace K 线数据下載工具 (修正版)
============================================================
安装必需的库...
依赖安装完成

可用的数据列表:
...
开始下載 K 线数据...
============================================================
下載 BTCUSDT 15m K 线数据...
成功下載 BTCUSDT 15m: 10000 行数据

数据列名: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
已保存到: downloaded_klines/BTCUSDT_15m_10000.csv

BTC USDT 数据统计:
  行数: 10000
  列数: 6
  日期范围: 2025-09-14 09:30:00 到 2025-12-27 13:15:00

前 5 行数据:
  timestamp    open    high    low   close volume
0  2025-09-14 09:30:00  42100  42105  42090  42105   0.05
1  2025-09-14 09:45:00  42105  42150  42090  42140   0.12
2  2025-09-14 10:00:00  42140  42180  42120  42160   0.08
3  2025-09-14 10:15:00  42160  42200  42140  42190   0.15
4  2025-09-14 10:30:00  42190  42220  42170  42210   0.10

数据已保存，可以用于后续分析
```

### 第二步：结合 trainer 处理

```python
import pandas as pd
from trainer import preprocess_and_label

# 读取下載的数据
df = pd.read_csv('downloaded_klines/BTCUSDT_15m_10000.csv')

# 用 trainer 处理
result = preprocess_and_label(df)

# 保存结果
result.to_csv('processed_BTCUSDT_15m.csv', index=False)
```

---

## 最常见的错误上报

### 404 错误

```
404 Client Error. Entry Not Found for url: 
https://huggingface.co/datasets/zongowo111/cpb-models/resolve/main/klines_binance_us/BTCUSDT_15m.csv
```

**原因：** 路径错误 (klines_binance_us/ 多了一个 '_')

**解决：** 使用修正版脚本 `colab_hf_corrected.py`

### 列名不匹配错误

```
警告: 数据格式不符。现有列: [...]
```

**原因：** CSV 的列名不是预期的

**解决：** 检查 CSV 文件的实际列名，可以修改脚本中的 `required_columns` 列表

---

## 下一步

1. 在 Colab 中执行 `colab_hf_corrected.py` 下載数据
2. 使用 trainer 项目处理数据
3. 结合床上五阶段算法进行反轉检测
4. 生成训练数据进行模型学习

---

## 参考

- HuggingFace 数据集： https://huggingface.co/datasets/zongowo111/cpb-models
- 你的 trainer 项目： https://github.com/caizongxun/trainer
- swing-reversal-prediction 项目： https://github.com/caizongxun/swing-reversal-prediction
