# PHASE1 涨跌幅阈值过滤 - 滤掉假信号

---

## 核子问题

之前的方向改变法只搬查是否改变方向。
但实际上有许外多假反转：

```
反转模式 A: 仃上 下 仃上 ↔ 有效
反转模式 B: 仃上 下 仃上 下 过【不算】
反转模式 C: 仃上 下 仃上 → 厚写【假】
```

啊，你圆的不一样。

---

## 解决方案：涨跌幅阈值

我们不是只是标记方向改变，而是要求反转后必须悬质有辭度的涨跌幅。

```python
例子：
低点反转：low=87000
之后最高=87435
涨幅 = (87435-87000)/87000 * 100 = 0.5%

如果 amplitude_threshold=0.5%, → 有效
如果 amplitude_threshold=1.0%, → 无效（傍不足足）
```

---

## 需要的参数

### --lookback

向后看訊根K线。
推荐 **10** （避免当到不必要的上下波动）

### --amplitude_threshold

涨跌幅阈值 (%).

| 值 | 效果 | 备注 |
|-----|------|------|
| 0.3% | 越多的反转点 | 一些是假信号 |
| 0.5% | **推荐** | 偶然是假信号 |
| 1.0% | 越少的反转点 | 达民都是有效 |
| 2.0% | 最有效 | 但业会遗漏小反转 |

### --confirm_local

是否验证局部高低点。
不推荐 (consume前已经过滤掉了）

---

## 强碎使用方法

### 步骤1: 下载脚本

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_amplitude_filter.py -O phase1.py
```

### 步骤2: 执行 (不同的参数)

```bash
# 最次优推谐（一上来筛选假信号）
!python3 phase1.py --lookback 10 --amplitude_threshold 0.5 --confirm_local False

# 对比：更有效（可能遗漏小反转）
!python3 phase1.py --lookback 10 --amplitude_threshold 1.0 --confirm_local False

# 对比：捕捉更多（更容易有假信号）
!python3 phase1.py --lookback 10 --amplitude_threshold 0.3 --confirm_local False
```

---

## 结果对比

### 【旧方法】 amplitude_threshold=无（没有过滤）

```
两个都是反转点：
- A: 上下上 (+ 0.1% 涨幅) ✔ 有效
- B: 上下上 (+ 0.01% 涨幅) ❌ 假信号
```

的会标记 280 个反转点！

### 【新方法】 amplitude_threshold=0.5%

```
两个反转点：
- A: 上下上 (+ 0.5% 涨幅) ✔ 保留
- B: 上下上 (+ 0.01% 涨幅) ❌ 滤掉
```
只标记 50-80 个反转点！
馬上滤掉了假信号。

---

## 你需要的是

1. 可筛选出真正有效的反转点。
2. 不需要手动标记。
3. 自动涨跌幅计算并滤掉窗。

---

## 啊，Colab怎么借？

```bash
# Cell 1
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_amplitude_filter.py -O phase1.py

# Cell 2 (多个)
!python3 phase1.py --lookback 10 --amplitude_threshold 0.5 --confirm_local False
!python3 phase1.py --lookback 10 --amplitude_threshold 1.0 --confirm_local False
!python3 phase1.py --lookback 10 --amplitude_threshold 0.3 --confirm_local False
```

每执行一行，自动生成 CSV + PNG。

按会次纷边加一个参数错误吗！

---

## 来免号

现在都是自动：

✓ 自动检测方向改变
✓ 自动计算涨跌幅
✓ 自动滤掉假信号
✓ 自动生成报告

你只需调一个参数： amplitude_threshold

---

## 下一步

优化了反转点标记后，你可以：

1. 下载最满意的CSV文件
2. 进行PHASE2：特征提取
3. PHASE3：模型训练
4. PHASE4：规则生成
