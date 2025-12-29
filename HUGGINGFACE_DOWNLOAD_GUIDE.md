# HuggingFace K 线数据下载完整指南

## 成功！🌟

已经成功实现了自动下载不的HuggingFace K线数据功能。

---

## 快速开始

### 在 Colab 中执行

```python
# Step 1: 重啟運行時
# 運行時 → 重啟運行時

# Step 2: 克隆倉庫
!rm -rf /content/swing-reversal-prediction 2>/dev/null || true
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction

# Step 3: 執行下載脚本
!python colab_hf_working.py
```

---

## 脚本象征

### 正式脚本：`colab_hf_working.py`

**功能：**
- 自动下载 HuggingFace K线数据
- 自动标准化列名
- 保存到本地 CSV 文件
- 支持所有 20 个交易对
- 支持 15m 和 1h 两个时间周期

**提供者：**
- zongowo111 (GitHub)
- [HuggingFace 数据集](https://huggingface.co/datasets/zongowo111/cpb-models)

**最低要求：**
- Python 3.7+
- pandas, numpy
- datasets, huggingface-hub

---

## 数据规格

### 数据集位置

```
https://huggingface.co/datasets/zongowo111/cpb-models
```

### 文件路径

```
klines_binance_us/{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv
```

**示例：**
- BTCUSDT 15分钟: `klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv`
- ETHUSDT 1小时: `klines_binance_us/ETHUSDT/ETHUSDT_1h_binance_us.csv`

### 数据列

**原始列（HuggingFace）：**
```
open_time, open, high, low, close, volume, close_time
```

**标准化后（脚本输出）：**
```
timestamp, open, high, low, close, volume
```

### 数据特点

- **每个文件：** 10,000 行 K线数据
- **时间范围：** 最近 3-12 个月
- **数据源：** Binance US
- **位数：** 开針价、最高价、最低价、收重价、成交量

---

## 支持的交易对

### 20 个主流互项目

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

### 时间周期

- **15m** - 15分钟 K线
- **1h** - 1小时 K线

---

## 成功的执行示例

```
============================================================
HuggingFace K 线数据下載工具 (工作版)
============================================================
安装必需的库...
依赖安装完成

可用的数据列表:
...

开始下載 K 线数据...
============================================================
下载 BTCUSDT 15m K 线数据...
  路径: klines_binance_us/BTCUSDT/BTCUSDT_15m_binance_us.csv
  ✓ 成功下载: 10000 行数据
  原始列名: ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
  ✓ 列名已标准化: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
  保存到: downloaded_klines/BTCUSDT_15m_10000.csv

✓ 成功！

BTC USDT 数据统计:
  行数: 10000
  列数: 6
  列名: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
  日期范围: 2025-09-14 09:30:00 到 2025-12-27 13:15:00

前 5 行数据:
             timestamp       open       high        low      close   volume
0  2025-09-14 09:30:00  116025.32  116054.89  116025.32  116054.89  0.02515
1  2025-09-14 09:45:00  116059.68  116126.19  115865.35  116125.87  0.05250
2  2025-09-14 10:00:00  116104.30  116104.30  116101.96  116101.96  0.00018
3  2025-09-14 10:15:00  116101.96  116101.96  116101.96  116101.96  0.00000
4  2025-09-14 10:30:00  116102.53  116102.53  116102.53  116102.53  0.00006

✓ 数据已保存，可以用于后续分析
```

---

## 文件位置

执行脚本后，数据会保存到：

```
downloaded_klines/
├─ BTCUSDT_15m_10000.csv
├─ BTCUSDT_1h_10000.csv
├─ ETHUSDT_15m_10000.csv
├─ ETHUSDT_1h_10000.csv
└─ ...
```

---

## 下載到本地

### 从 Colab 下載

```python
from google.colab import files

# 下載单个文件
files.download('downloaded_klines/BTCUSDT_15m_10000.csv')

# 或者下載整个文件太苦，可以先压缩：
import os
os.system('cd downloaded_klines && zip -r ../klines_data.zip *.csv')
files.download('klines_data.zip')
```

---

## 下一步：整合 trainer 项目

### 1. 读取下載的数据

```python
import pandas as pd

df = pd.read_csv('downloaded_klines/BTCUSDT_15m_10000.csv')
print(df.head())
print(df.info())
```

### 2. 接入 trainer 预处理

```python
from trainer import preprocess_and_label

# 用 trainer 处理
result = preprocess_and_label(df)

# 保存结果
result.to_csv('processed_BTCUSDT_15m.csv', index=False)
```

### 3. 批量下載所有数据

```python
import pandas as pd
from colab_hf_working import HuggingFaceKlinesDownloader

downloader = HuggingFaceKlinesDownloader()
downloader.install_dependencies()

# 下載诌个交易对
trading_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']

for pair in trading_pairs:
    df = downloader.fetch_and_combine(pair, '15m', 10000)
    if df is not None:
        print(f"{pair}: {len(df)} 行")
```

---

## 粗故排除

### 问题 1: 404 错误

错误：
```
404 Client Error. Entry Not Found
```

解决：
- 确保使用最新的 `colab_hf_working.py`
- 确保正确的路径格式：`klines_binance_us/{PAIR}/{PAIR}_{TIMEFRAME}_binance_us.csv`

### 问题 2: 列名错误

错误：
```
警告: 列名预期不符
```

解决：
- 脚本已自动处理列名映射
- `open_time` 自动会转换为 `timestamp`

### 问题 3: 网络错误

错误：
```
Network error, timeout
```

解决：
- 检查互联网连接
- 重试下載
- 需要时转换到不同的时間

---

## 統计信息

### 总洛数据量

- **交易对：** 20 个
- **时间周期：** 2 个 (15m, 1h)
- **每漅数据：** 10,000 行
- **总瑞数据：** 20 × 2 × 10,000 = 400,000 行
- **总抬数需：** 约 200-300 MB

### 执行时间

- 单个交易对：约 10-20 秒
- 批量下載（20 个交易对）：约 5-10 分钟

---

## 提供者信息

**数据提供者：** [zongowo111 - HuggingFace](https://huggingface.co/zongowo111)

**数据集：** [zongowo111/cpb-models](https://huggingface.co/datasets/zongowo111/cpb-models)

**脚本引読：** swing-reversal-prediction GitHub 项目

**一次性不需要改一则：** 下載后数据不会变化，每次执行脚本都会加載最新的数据。

---

## GitHub 提交

- **工作脚本：** `colab_hf_working.py` [hash: 6433037]
- **编码修正：** 最旨简分三次修正
  1. 路径一：`klines_binance_us/`
  2. 路径二：`klines/`
  3. 路径三：`klines_binance_us/` (正确)
  4. 列名一：`open_time` -> `timestamp`

---

## 概述

有了该脚本，你已经可以：

✓ 自动下載任意数量的 HuggingFace K线数据
✓ 标准化数据列
✓ 一键合成 trainer 项目
✓ 用于反轉检测算法
✓ 用于模型训练

---

## 文档版本

**配置日期：** 2025-12-30
**最后更新：** 2025-12-30
**脚本版本：** Working v1.0
