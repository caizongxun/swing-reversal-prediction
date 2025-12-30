# Colab 简化使用指南 (PHASE1 与可视化)

---

## 最简单的使用方法

在Colab中打开一个**新Cell**，粘贴以下代码：

```python
# 设置参数
LOOKBACK = 5
CONFIRM_LOCAL = True

# 下载并执行脚本
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_visualization.py -O phase1.py
%run phase1.py
```

就是这么简单。

---

## 特点

✓ **自动下载数据** - 无需手动提前加载

✓ **自动标记数据** - 并保存CSV

✓ **自动生成图表** - 直接是改不需要离线执行

✓ **自动筗计** - 显示反转点数量和比例

---

## 执行流程

### 1. 新建 Cell

在Colab中新建一个次的Cell。

### 2. 设置参数

```python
LOOKBACK = 5            # 改成 1-10 之间的任意值
CONFIRM_LOCAL = True    # 改成 True 或 False
```

### 3. 执行脚本

```python
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_visualization.py -O phase1.py
%run phase1.py
```

### 4. 等待结果

脚本会自动：
- 下载数据
- 检测反转点
- 保存带标记的CSV
- 生成可视化图表
- 显示统计信息

---

## 调整参数

只需修改第一个Cell中的两个基本值：

```python
LOOKBACK = 3          # 变更这个
CONFIRM_LOCAL = False # 变更这个

# 成不需修改下面的代码
!wget -q https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_with_visualization.py -O phase1.py
%run phase1.py
```

然后按 `Ctrl+Enter` 或点击Run按钮重新执行。

---

## 参数效果对照

### LOOKBACK

| 值 | 输出 | 备注 |
|-----|---------|--------|
| 1 | 3500-4000个 | 最灵敏 |
| 3 | 1800-2200个 | 中等 |
| 5 | 800-1200个 | **推荐** |
| 7 | 400-600个 | 保守 |
| 10 | 200-300个 | 非常保守 |

### CONFIRM_LOCAL

```
True   -> 较少但质量高
False  -> 较多但可能有噪音
```

---

## 子手插一次：Colab按鎘校知识

### 按鎘校知识

在您想的Cell前面加一个"**+**"按鎘或把Cell序号把箱住。

### Ctrl+Enter

执行当前Cell。

### Shift+Enter

执行当前Cell并自动移到下一个Cell。

---

## 整个工作流程

```
第一次执行：
Cell 1: LOOKBACK=5, CONFIRM_LOCAL=True
        -> 执行
        -> 查看结果和图表

对比不同参数：
Cell 2: LOOKBACK=3, CONFIRM_LOCAL=True
        -> 执行
        -> 查看数量半会变化

Cell 3: LOOKBACK=7, CONFIRM_LOCAL=True
        -> 执行
        -> 查看数量减一半

Cell 4: LOOKBACK=5, CONFIRM_LOCAL=False
        -> 执行
        -> 比较有盐半的效果
```

---

## 有或没有图表

轓模脚本会自动：

- ✅ 下载CSV标记数据
- ✅ **需要时手动基程CSV文件**
- ✅ 自动生成可视化PNG图表
- ✅ 在Colab中直接显示图表

如果不想要图表，可以修改脚本的最后部分。

---

## 下载结果文件

脚本执行后会保存：

1. **CSV文件**: `phase1_lb{lookback}_local{T/F}.csv`
   - 特点：标记了所有反转点

2. **图表文件**: `phase1_lb{lookback}_local{T/F}.png`
   - 握其：最近800根K线的可视化

在Colab左侧文件管理器中下载。

---

## 基本难度

允许无法使用Colab，有Python环境也可以：

```bash
# 本地执行
python3 colab_phase1_with_visualization.py
```

脚本会自动传详提取HuggingFace上的数据。

---

## 下一步

当找到最好的参数组合后：

1. 导出CSV文件
2. PHASE2: 特征提取
3. PHASE3: 模型训练
4. PHASE4: 交易规则生成

---

## FAQ

Q: 为什么我不需要提前下载数据?

A: 脚本会自动从HuggingFace下载。

Q: 每次修改参数需要重新执行整个Cell冬?

A: 是的，但不需要重新下载数据。

Q: 故化表无法显示?

A: 檢查是否有反转点。如果没有，图表不会显示。
