# PHASE1 混合标记系统 - 趋势反转 + 横盘区域

---

## 核心思路

您提出的是非常高级的标记策略：

```
问题: 趋势反转点方法无法处理横盘
解决: 两阶段标记系统

Step 1: 判断市场模式
  if 最高-最低 < 0.5% within 30 bars → 横盘
  else → 趋势

Step 2: 根据模式标记
  趋势模式 → 标记反转点 (已有)
  横盘模式 → 标记压力/支撑区 (新增)
```

---

## 使用方法

### 下载脚本

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_hybrid_labeling.py -O phase1_hybrid.py
```

### 执行命令

#### 选项1: 混合模式 (推荐)

```bash
!python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode hybrid
```

自动：
- 检测趋势段 → 标记反转点
- 检测横盘段 → 标记支撑/压力

#### 选项2: 仅趋势模式

```bash
!python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode trend
```

只在趋势段标记反转点，忽略横盘。

#### 选项3: 仅横盘模式

```bash
!python3 phase1_hybrid.py --detect_mode sideways
```

只在横盘段标记支撑/压力，忽略趋势。

---

## 参数说明

### 核心参数

| 参数 | 说明 | 范例 | 默认值 |
|-----|------|------|--------|
| `--lookback` | 趋势判断周期 | 7 | 7 |
| `--amplitude_threshold` | 趋势反转最低幅度 | 0.656 | 0.656 |
| `--sideways_threshold` | 横盘判断阈值 (%) | 0.5 | 0.5 |
| `--window_size` | 横盘检测窗口 (根数) | 30 | 30 |
| `--detect_mode` | 检测模式 | hybrid/trend/sideways | hybrid |

### sideways_threshold 说明

```
在一个window内，如果:
(最高价 - 最低价) / 最低价 * 100 < threshold

则该区间为横盘

范例:
window内: high=88500, low=88000
ratio = (88500-88000)/88000*100 = 0.568%

if threshold=0.5% → 横盘
if threshold=1.0% → 趋势
```

### window_size 说明

检测横盘的滑动窗口大小

```
window_size=30  → 检查连续30根K线
           20  → 检查连续20根K线 (更敏感)
           40  → 检查连续40根K线 (更严格)
```

---

## 输出数据列

### CSV 新增列

| 列名 | 说明 | 范例 |
|-----|------|------|
| `Market_Mode` | 当前K线的市场模式 | Trend / Sideways |
| `Label_Type` | 标记类型 | Local_High / Local_Low / Resistance / Support / None |
| `Label_Value` | 标记价格 | 88500.5 |
| `Label_Strength` | 标记强度 | 幅度% 或 区间% |

### 标记类型解释

**趋势段标记：**
- `Local_High` - 卖点（下跌反转）
- `Local_Low` - 买点（上升反转）
- 附加信息：幅度% (反转后的涨跌幅)

**横盘段标记：**
- `Resistance` - 压力区 (最高价)
- `Support` - 支撑区 (最低价)
- 附加信息：区间% (最高-最低的百分比)

---

## 实际例子

### 您提到的图表

```
图: 0-100根  → 橫盤
    100-200根 → 趋势向上 → 有反转点
    200-300根 → 趋势向上 → 有反转点
    300-400根 → 趋势向下 → 有反转点
    400-500根 → 橫盤 → 有支撑/压力
    ...
```

**原来的方法：** 都标记为反转点，混乱
**混合方法：** 
- 趋势段: Local_High/Local_Low
- 橫盤段: Resistance/Support

---

## 调整建议

### 如果检测出太多横盘

```bash
# 提高横盘阈值（更难被判定为横盘）
!python3 phase1_hybrid.py --sideways_threshold 1.0 --window_size 30
```

### 如果检测出太少横盘

```bash
# 降低横盘阈值（更容易被判定为横盘）
!python3 phase1_hybrid.py --sideways_threshold 0.3 --window_size 20
```

### 如果反转点质量仍然不好

```bash
# 增加幅度要求
!python3 phase1_hybrid.py --amplitude_threshold 1.0
```

---

## 输出可视化

生成两个图表：

1. **上图：** K线价格 + 标记点
   - 灰线: 收盘价
   - 蓝色背景: 横盘区域
   - 绿色三角: 买点 (Local_Low)
   - 红色三角: 卖点 (Local_High)
   - 橙色方块: 压力区 (Resistance)
   - 紫色方块: 支撑区 (Support)

2. **下图：** 市场模式演化
   - 0 = Trend (趋势)
   - 1 = Sideways (横盘)

---

## 快速对比

执行多次对比不同参数：

```bash
# 执行1: 标准混合
!python3 phase1_hybrid.py --detect_mode hybrid --sideways_threshold 0.5 --window_size 30

# 执行2: 更敏感的横盘
!python3 phase1_hybrid.py --detect_mode hybrid --sideways_threshold 0.3 --window_size 20

# 执行3: 严格模式
!python3 phase1_hybrid.py --detect_mode hybrid --sideways_threshold 1.0 --amplitude_threshold 1.0
```

每个会生成不同的CSV+PNG，方便对比。

---

## 常见问题

Q: Trend 和 Sideways 的比例大概是多少？

A: 通常是 70% Trend / 30% Sideways，但取决于市场状况和参数。

Q: 如果 sideways_threshold 设太低会怎样？

A: 大部分K线都会被判为横盘，反转点会很少。

Q: Resistance 和 Support 是否会重叠？

A: 不会，系统会合并相邻的横盘段。

Q: 我想要更多的反转点？

A: 降低 amplitude_threshold (例如 0.3 而不是 0.656)，但质量会下降。

Q: 支撑/压力区的数量为什么比反转点少？

A: 因为系统会合并相邻的横盘段，所以一个长横盘段只标记一对 S/R。

---

## 下一步

得到满意的标记后：

1. 下载 CSV 文件
2. 进行 PHASE2: 特征提取
3. PHASE3: 模型训练
4. PHASE4: 交易规则生成

---

## 总结

这个混合系统：

✓ 自动检测市场模式
✓ 趋势段用反转点
✓ 横盘段用支撑/压力
✓ 完全自动化，不需要手动标记
✓ 可调参数优化效果
