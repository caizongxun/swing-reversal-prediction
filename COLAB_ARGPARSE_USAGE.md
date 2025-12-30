# Colab ArgParse 参数方式

---

## 最简单的使用方式

在Colab Cell中执行以下命令：

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_argparse_version.py -O phase1.py
!python3 phase1.py --lookback 7 --confirm_local True
```

就这样。完成。

---

## 参数使用

### 示例1: 改lookback

```bash
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_argparse_version.py -O phase1.py
!python3 phase1.py --lookback 3 --confirm_local True
```

### 示例2: 改confirm_local

```bash
!python3 phase1.py --lookback 5 --confirm_local False
```

### 示例3: 两个都改

```bash
!python3 phase1.py --lookback 7 --confirm_local False
```

### 示例4: 使用默认参数

```bash
!python3 phase1.py
```

使用默认参数 (lookback=5, confirm_local=True)

### 示例5: 自定义输出文件

```bash
!python3 phase1.py --lookback 5 --confirm_local True --output my_result.csv
```

---

## 参数说明

| 参数 | 说明 | 范围 | 默认值 |
|-----|------|------|--------|
| `--lookback` | 向后看N根K线 | 1-10 | 5 |
| `--confirm_local` | 验证局部高低点 | True/False | True |
| `--output` | 输出文件名 | 任意 | 自动生成 |

---

## 结果对照表

### Lookback影响

```
lookback=3  -> 1800-2200 个反转点
lookback=5  -> 800-1200 个反转点 (推荐)
lookback=7  -> 400-600 个反转点
lookback=10 -> 200-300 个反转点
```

### Confirm_local影响

```
confirm_local=True  -> 质量高，噪音少
confirm_local=False -> 数量多，可能有噪音
```

---

## 快速对比

想要快速对比多个参数？逐个执行以下命令：

```bash
# 执行1
!python3 phase1.py --lookback 3 --confirm_local True

# 执行2
!python3 phase1.py --lookback 5 --confirm_local True

# 执行3
!python3 phase1.py --lookback 7 --confirm_local True

# 执行4
!python3 phase1.py --lookback 5 --confirm_local False
```

每个命令都会生成不同的CSV和图表文件，方便对比。

---

## 输出说明

### 参数确认

```
[参数设置]
  --lookback 7
  --confirm_local True
```

这里会显示您设置的参数。

### 检测结果

```
总反转点数: 600
  - 局部低点 (买入点): 300
  - 局部高点 (卖出点): 300

反转点占比: 6.00%
平均间距: 16 根K线
```

### 生成文件

- `phase1_lb7_localT.csv` - 标记好的数据
- `phase1_lb7_localT.png` - 可视化图表

---

## 下载结果

执行完毕后，在Colab左侧文件管理器中可以看到生成的文件。

点击文件名旁的三点菜单 → "下载"即可。

---

## 下一步

当找到最好的参数后：

1. 下载标记好的CSV文件
2. 进行PHASE2: 特征提取
3. PHASE3: 模型训练
4. PHASE4: 交易规则生成

---

## 常见问题

Q: 为什么要--lookback和--confirm_local?

A: 这两个参数直接影响标记点的数量和质量。需要多次调整找到最优值。

Q: 哪个参数组合最好?

A: 推荐 `--lookback 5 --confirm_local True`，这是平衡点。

Q: 我想要更多的反转点。

A: 试试 `--lookback 3 --confirm_local False`

Q: 我想要更少的反转点。

A: 试试 `--lookback 7 --confirm_local True` 或 `--lookback 10 --confirm_local True`

Q: 执行时间需要多长?

A: 首次执行（需要下载数据）约1-2分钟。之后约10-30秒。

---

## 总结

这是最灵活的参数调整方式：

✓ 支持完整的--flag参数
✓ 参数验证
✓ 自动输出文件
✓ 一条命令执行
✓ 结果清晰可见
