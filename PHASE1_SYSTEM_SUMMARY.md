# PHASE1 标记系统完整总结

---

## 您提出的问题与解决方案

### 问题1: 假信号太多
**您说:** 即使参数调到最大，仍有许多假信号

**解决:** 加入 `--amplitude_threshold` 参数
- 反转后必须有足够的涨跌幅才算有效
- 默认 0.656% → 可调整到 0.5% 或 1.0%
- 效果: 280个反转点 → 84个有效反转点

### 问题2: 横盘段没有处理
**您说:** 横盘波段可能要用压力区和阻力区标记

**解决:** 创建混合标记系统
- Step 1: 自动检测市场模式 (趋势 vs 横盘)
- Step 2: 趋势段用反转点标记
- Step 3: 横盘段用支撑/阻力区标记

---

## 三个脚本对比

### 脚本1: 基础反转点检测
**文件:** `colab_phase1_with_amplitude_filter.py`

**用法:**
```bash
!python3 phase1.py --lookback 7 --amplitude_threshold 0.656 --confirm_local True
```

**输出列:**
- Reversal_Label (0/1)
- Reversal_Type (Local_High / Local_Low / None)
- Reversal_Amplitude (涨跌幅%)

**适用场景:**
- 只关心趋势反转点
- 市场始终在趋势中

**缺点:** 完全忽视了横盘段

---

### 脚本2: 混合标记系统 ← 推荐给您
**文件:** `colab_phase1_hybrid_labeling.py`

**用法:**
```bash
# 混合模式 (推荐)
!python3 phase1_hybrid.py --lookback 7 --amplitude_threshold 0.656 --detect_mode hybrid --sideways_threshold 0.5

# 仅趋势
!python3 phase1_hybrid.py --detect_mode trend

# 仅横盘
!python3 phase1_hybrid.py --detect_mode sideways
```

**输出列:**
- Market_Mode (Trend / Sideways)
- Label_Type (Local_High / Local_Low / Resistance / Support / None)
- Label_Value (价格)
- Label_Strength (幅度% 或 区间%)

**核心逻辑:**

```
步骤1: 检测市场模式
  for each 30-bar window:
    if (high - low) / low < 0.5% → 这是横盘
    else → 这是趋势

步骤2: 根据模式标记
  if 趋势 → Local_High / Local_Low (需要幅度验证)
  if 横盘 → Resistance (最高) / Support (最低)

步骤3: 合并相邻横盘段
  防止支撑/阻力过度碎片化
```

**优点:**
- 自动检测市场模式
- 趋势段和横盘段都能处理
- 参数可调，灵活性高
- 完全自动化，不需手动标记

**参数调整:**

| 参数 | 范围 | 效果 |
|-----|------|------|
| `--sideways_threshold` | 0.3-1.5 | 越小越容易判定为横盘 |
| `--window_size` | 20-40 | 越大检测越粗糙 |
| `--amplitude_threshold` | 0.3-1.5 | 只影响趋势段反转点质量 |

---

## 您的图表分析

### 观察结果
```
您的截图:
- lookback=8, amplitude>=0.65%
- 最近800根K线中: 84个有效反转
- 38个Local Low, 46个Local High

可视化观察:
- 0-180 根:   明显的横盘区 (无反转信号)
- 180-300根:  强势上升趋势 (密集反转信号)
- 300-400根:  下跌调整 (反转信号较少)
- 400-500根:  又开始横盘 (无反转信号)
- ...
```

### 如果用混合脚本会怎样

**执行:**
```bash
!python3 phase1_hybrid.py --lookback 8 --amplitude_threshold 0.65 --detect_mode hybrid
```

**预期输出:**
- 0-180根: Market_Mode = Sideways → 标记为 Resistance/Support
- 180-300根: Market_Mode = Trend → 标记为 Local_High/Local_Low
- 300-400根: Market_Mode = Trend → 标记为 Local_High/Local_Low
- 400-500根: Market_Mode = Sideways → 标记为 Resistance/Support

**结果:** CSV中同时有趋势反转点和横盘支撑/阻力区，完整！

---

## 推荐方案 (给您)

### 第一步: 快速验证

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_hybrid_labeling.py -O phase1_hybrid.py

# 用您之前的参数，加上混合模式
!python3 phase1_hybrid.py --lookback 8 --amplitude_threshold 0.65 --detect_mode hybrid
```

### 第二步: 查看结果

- 打开 `phase1_hybrid_lb8_amp0.65_hybrid.csv`
- 查看 Market_Mode 列
- 查看 Label_Type 列
- 应该同时有 Trend/Sideways 和 Local_High/Resistance

### 第三步: 细调参数

**如果横盘检测得太少:**
```bash
!python3 phase1_hybrid.py --sideways_threshold 0.3 --window_size 20
```

**如果横盘检测得太多:**
```bash
!python3 phase1_hybrid.py --sideways_threshold 1.0 --window_size 40
```

**如果反转点质量还不够:**
```bash
!python3 phase1_hybrid.py --amplitude_threshold 1.0
```

### 第四步: 下载满意的结果

选择最满意的 CSV，进入 PHASE2

---

## 技术细节 (可选阅读)

### 横盘检测算法

```python
def detect_sideways_segments(df, window_size=30, threshold=0.5):
    sideways_segments = []
    
    # 滑动窗口
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        high = window['high'].max()
        low = window['low'].min()
        
        # 计算幅度比例
        ratio = ((high - low) / low) * 100
        
        # 如果幅度小于阈值 → 横盘
        if ratio < threshold:
            sideways_segments.append({
                'start': i,
                'end': i + window_size,
                'resistance': high,
                'support': low,
                'range_percent': ratio
            })
    
    # 合并相邻横盘段 (避免碎片化)
    merged = merge_adjacent_segments(sideways_segments)
    return merged
```

### 反转点验证算法

```python
def check_amplitude(idx, direction_type, threshold, look_ahead):
    reversal_price = df.iloc[idx]['close']
    max_idx = min(idx + look_ahead, len(df) - 1)
    
    if direction_type == 'up':  # Low反转，之后需要涨
        max_price = df.iloc[idx:max_idx+1]['high'].max()
        amplitude = ((max_price - reversal_price) / reversal_price) * 100
    else:  # High反转，之后需要跌
        min_price = df.iloc[idx:max_idx+1]['low'].min()
        amplitude = ((reversal_price - min_price) / reversal_price) * 100
    
    return amplitude, amplitude >= threshold
```

---

## 常见问题

Q: Sideways_threshold 应该设多少?
A: 根据您的市场。BTC 15m 建议 0.5%。

Q: Window_size 越大越好吗?
A: 不是。太大会漏掉短期横盘。30 是平衡值。

Q: 为什么支撑/阻力点数比反转点少?
A: 因为长横盘段会被合并成一对 S/R，所以数量少。

Q: 我能同时运行两个脚本吗?
A: 可以。用不同的 --output 参数保存，然后比较。

Q: 下一步是什么?
A: PHASE2 特征提取。我们会提取反转点前的技术面数据。

---

## 文件清单

| 文件 | 功能 | 何时用 |
|-----|------|------|
| `colab_phase1_with_amplitude_filter.py` | 基础反转点检测 | 简单场景 |
| `colab_phase1_hybrid_labeling.py` | 混合标记系统 | 推荐给您 |
| `PHASE1_AMPLITUDE_FILTER_GUIDE.md` | 反转点参数指南 | 调参参考 |
| `PHASE1_HYBRID_LABELING_GUIDE.md` | 混合系统详细指南 | 深度了解 |
| `PHASE1_QUICK_START.md` | 快速对比指南 | 快速决策 |

---

## 总结

您提出的混合策略非常聪慧：

- 趋势段：反转点明确，用幅度过滤
- 横盘段：没有方向，用支撑/阻力标记
- 两种标记方式融合在一个 CSV 中

现在系统可以完全自动化处理这两种情况。

下一个阶段：PHASE2 特征提取。
