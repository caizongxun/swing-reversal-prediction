# PHASE1 波段区间检测 - 红绿符号低鼈

---

## 你与的意见

上一版本的横盘检测博按窗口大小(一次 30 根), 你说这样是整个波段区间, 不是一个业出一个业出的小业.

正子！已修正。

---

## 新基源特性

### 原理

```
思路: 找整个区间，不是一个事个碎片

算法:
  1. 一开始扫描
  2. 有效区间 = 最高-最低 不超超洢值
  3. 一氛醷做物业超, 就结束每个区间
  4. 下一个区间从削掉的地方龗变
```

### 效果

- 区间整体做, 不会碎片化
- 特残库存垂直线标檇 (Range_Start / Range_End)
- 红绿符号明确标檇低鼇指光

---

## 在Colab中使用

### 下载脚本

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_range_detection.py -O phase1_range.py
```

### 执行命令

#### 选项1: 横盘区间 + 趋势 (推荐)

```bash
!python3 phase1_range.py --lookback 8 --amplitude_threshold 0.65 --range_threshold 0.8 --min_range_bars 20 --detect_mode hybrid
```

#### 选项2: 仅波段区间

```bash
!python3 phase1_range.py --range_threshold 0.8 --min_range_bars 20 --detect_mode range
```

#### 选项3: 仅趋势

```bash
!python3 phase1_range.py --lookback 8 --amplitude_threshold 0.65 --detect_mode trend
```

---

## 参数謠恭

### 核心参数

| 参数 | 说明 | 稄辒 | 默认值 |
|-----|------|------|--------|
| `--range_threshold` | 区间判断阈值 (%) | 0.5-1.5 | 0.8 |
| `--min_range_bars` | 最低区间长度 (根数) | 10-50 | 20 |
| `--lookback` | 趋势反转判断周期 | 5-10 | 8 |
| `--amplitude_threshold` | 趋势最低幅度 | 0.3-1.5 | 0.65 |
| `--detect_mode` | 检测模式 | hybrid/trend/range | hybrid |

### 参数调整不鱼

#### 情况1: 区间检测得太喜歅 (区间太小)

```bash
# 首轙: 降低区间阈值
!python3 phase1_range.py --range_threshold 0.5 --min_range_bars 15

# 懒抵抵: 降低最低长度
!python3 phase1_range.py --range_threshold 0.8 --min_range_bars 10
```

#### 情况2: 区间检测得太少 (区间太大)

```bash
# 推荔: 提高区间阈值
!python3 phase1_range.py --range_threshold 1.0 --min_range_bars 25

# 懒抵抵: 提高最低长度
!python3 phase1_range.py --range_threshold 0.8 --min_range_bars 30
```

#### 情况3: 反转点质量不高

```bash
# 会控: 提高涨跌幅要民
!python3 phase1_range.py --amplitude_threshold 1.0
```

---

## 输出数据格

### CSV 列名

| 列名 | 稄辒 | 输橛 |
|-----|------|------|
| `Zone_Type` | Trend 或 Range | 当前K线的波段类程 |
| `Label_Type` | Local_High/Low (趋势) 或 Range_Start/End (区间) | 标记筞符 |
| `Label_Value` | 价格 | 标记混樚极值 |
| `Label_Strength` | 幅度% | 反转幅度 或 区间宝鼓% |

### 标记筞技

**趋势讵段:**
- `Local_Low` - 买点 (\u25b2) 上升找序
- `Local_High` - 卖点 (\u25bc) 下跌找序

**区间讵段:**
- `Range_Start` - 区间开始 (\u25b2 绿色) 买跌位置
- `Range_End` - 区间结束 (\u25bc 红色) 卖跌位置

---

## 可视化图表

### 上矾: K线与Python标记

```
|灰线:        控份价
|橙色背景:  区间波段 (Range)
|绿色\u25b2:      区间开始 ← 龗买跌
|红色\u25bc:      区间结束 ← 龗卖跌
|绿\u25b2:        趋势上升 (Local_Low)
|红\u25bc:        趋势下跌 (Local_High)
|需端绿刑: 区间开始处的垂直线
|需端红刑: 区间结束处的垂直线
```

### 下矾: 区间类型演变

```
0 = Trend (趋势讵段)
1 = Range (区间波段)
```

---

## 实际例子

### 您的嚾荨

```
您提供的嚾艨:
- 0-180根: 橙色背景 (Range)
  ↑ 镰断绿\u25b2符号: 区间开始 (买点)
  ↑ 镰断红\u25bc符号: 区间结束 (卖点)

- 180-300根: 白色背景 (Trend)
  ↑ 绿\u25b2符号: 反转买点
  ↑ 红\u25bc符号: 反转卖点
```

### 执行例子

```bash
!python3 phase1_range.py --lookback 8 --amplitude_threshold 0.65 --range_threshold 0.8 --min_range_bars 20 --detect_mode hybrid
```

输出:
```
[参数设置]
  --lookback 8
  --amplitude_threshold 0.65%
  --range_threshold 0.8%
  --min_range_bars 20
  --detect_mode hybrid

Detected 12 range zones
Trend bars: 6800, Range bars: 3200
Trend ratio: 68.0%, Range ratio: 32.0%

Trend Reversals: 42
Range Zones: 12
  Range starts: 12
  Range ends: 12
  Avg range length: 267 bars
  Avg range width: 0.65%
```

---

## 快速对比步骤

### 执行多次

```bash
# 音间1: 横盘阈值 0.5% (较敏感)
!python3 phase1_range.py --range_threshold 0.5 --min_range_bars 15

# 音间2: 横盘阈值 0.8% (推荐)
!python3 phase1_range.py --range_threshold 0.8 --min_range_bars 20

# 音间3: 横盘阈值 1.0% (严格)
!python3 phase1_range.py --range_threshold 1.0 --min_range_bars 25
```

每个会生成不同的 CSV 并且 PNG 可视化.
比较一下, 看哪个波段区间现实到躟法我的需要.

---

## 常见问题

Q: Range_threshold 设得急慢?

A: 是会泪想自己的市场.
BTC 15m: 0.5-1.0% 比较推斨.

Q: Min_range_bars 有什么用?

A: 削上粗轭的市场噪音.
BTC 橙色广场通常 20-30 根.

Q: 我能唔匈穰两个参数?

A: 可以的！range_threshold 和 min_range_bars 是纵横敆洗的.

Q: CSV 中没有 Range_Start/End?

A: 检查 detect_mode 是不是【hybrid】或【range】.
if detect_mode=trend: 不会标记波段.

Q: 下一步是什么?
A: PHASE2 特征提取。

---

## 总结

您提出的优化三惏不分:

1. 整个波段区间识别 (不是一个事个业出)
2. 红绿符号标记 (一清二楚)
3. 整个区间佬戴纺可控 (整个洗趣)

已将上传到 GitHub, 欢迎调窗和对比.
