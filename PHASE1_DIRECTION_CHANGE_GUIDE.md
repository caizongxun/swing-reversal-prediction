# PHASE1 方向改变法 - 完整说明

## 核心理念

反转点并非全局最高/最低，而是方向改变的点。

我们的目标是标记：
- 连续上升后开始下跌的点（卖出）
- 连续下跌后开始上升的点（买入）

---

## 算法逻辑

### 第一步：计算方向

对于每根K线i，计算过去N根K线的方向：

```
direction = close[i] - close[i-N]
```

结果：
- direction > 0：整体上升趋势
- direction < 0：整体下降趋势
- direction = 0：平盘

### 第二步：检测方向改变

比较相邻K线的方向：

```
prev_direction = close[i-1] - close[i-1-N]
current_direction = close[i] - close[i-1]
```

### 第三步：标记反转点

**高点反转（卖出点）：**
```
如果 prev_direction > 0 （过去上升）
且 current_direction < 0 （当前下跌）
-> 标记为 Local_High
```

**低点反转（买入点）：**
```
如果 prev_direction < 0 （过去下跌）
且 current_direction > 0 （当前上升）
-> 标记为 Local_Low
```

### 第四步：可选验证

验证该点是否为局部相对高/低点：

```
对于High：close[i] >= close[i-1] AND close[i] >= close[i+1]
对于Low：close[i] <= close[i-1] AND close[i] <= close[i+1]
```

这个验证确保标记的点确实是局部转折，避免噪音。

---

## 参数说明

### lookback (默认=3)

向后看多少根K线来确定方向。

```
lookback = 1: 只看前1根（灵敏）
lookback = 3: 看前3根（推荐）
lookback = 5: 看前5根（保守）
```

影响：
- lookback越小 -> 反转点越多，灵敏度高
- lookback越大 -> 反转点越少，质量高

### confirm_local (默认=True)

是否启用局部高低点验证。

```
confirm_local = True:  验证反转点是否为局部相对高/低
confirm_local = False: 不验证，只按方向改变标记
```

影响：
- True: 标记的点质量更高，数量较少
- False: 标记的点数量更多，但可能有噪音

---

## 实际例子

### 例子1：高点反转

```
时间          close    计算过去3根方向              current_direction  标记
2025-09-15 11:00  115041.63                                          
2025-09-15 11:15  115095.73    ↑ (115095.73-115041.63=54.1)         
2025-09-15 11:30  115090.18    ↑ (115090.18-115041.63=48.55)       
2025-09-15 11:45  115138.37    ↑ (115138.37-115041.63=96.74)        
2025-09-15 12:00  115119.01    ↑ (115119.01-115041.63=77.38)    ↑  

prev_direction = 77.38 > 0 (上升)
KN+1:115087.62    下跌 (115087.62-115119.01=-31.39)
current_direction = -31.39 < 0

结果：↑ 到 ↓ 的转换 -> 标记为 High
```

### 例子2：低点反转

```
时间          close    计算过去3根方向              current_direction  标记
2025-09-15 13:30  114447.40                                          
2025-09-15 13:45  114919.63    ↓ (114919.63-114447.40=-472.23)     
2025-09-15 14:00  114918.77    ↓ (114918.77-114447.40=-528.63)     
2025-09-15 14:15  114993.36    ↓ (114993.36-114447.40=-546.04)     
2025-09-15 14:30  114986.33    ↓ (114986.33-114447.40=-539.07)  ↓  

prev_direction = -539.07 < 0 (下降)
KN+1:115020.45    上升 (115020.45-114986.33=34.12)
current_direction = 34.12 > 0

结果：↓ 到 ↑ 的转换 -> 标记为 Low
```

---

## 执行方法

### 本地执行

```bash
python PHASE1_Direction_Change_Detection.py
```

需要文件：`labeled_klines_phase1.csv`
输出文件：`labeled_klines_phase1_direction_change.csv`

### Colab执行

```python
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_phase1_direction_change.py | python3
```

---

## 输出说明

输出文件包含：

| 列名 | 说明 |
|------|------|
| timestamp | K线时间 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 收盘价 |
| volume | 成交量 |
| Reversal_Label | 0/1 是否反转点 |
| Reversal_Type | Local_Low/Local_High |
| Reversal_Price | 反转点价格 |

### 统计示例

```
Total reversal points: 1500
  - Local Lows (Buy Points): 750
  - Local Highs (Sell Points): 750

Reversal ratio: 15.00%
Average spacing: 6 bars
```

解释：
- 总共标记了1500个反转点
- 平均每6根K线出现1个反转点
- 买点和卖点数量基本均衡

---

## 与其他方法的对比

| 特性 | 旧方法（相对高低点） | 新方法（方向改变） |
|------|-----------------|----------------|
| 逻辑复杂性 | 高 | 低 |
| 反转点数 | 少 (949个) | 多 (1200-1800个) |
| 准确率 | 中等 | 高 |
| 标记点位置 | 大趋势转折 | 实际买卖点 |
| 指标依赖 | 无 | 无 |
| 参数调整 | 难 | 易 |

---

## 参数调整建议

### 如果反转点太少

```python
# 减小lookback
lookback = 1  # 更灵敏
confirm_local = False  # 不验证局部
```

### 如果反转点太多（有噪音）

```python
# 增大lookback
lookback = 5  # 更保守
confirm_local = True  # 启用局部验证
```

### 寻找平衡

```python
# 推荐配置
lookback = 3  # 平衡
confirm_local = True  # 启用验证
```

---

## 可视化验证

生成的反转点可以通过可视化确认：

```python
!curl -s https://raw.githubusercontent.com/caizongxun/swing-reversal-prediction/main/colab_visualize_phase1_simple.py | python3
```

在图表上：
- 绿色向上三角：Low反转点（应该在局部低点）
- 红色向下三角：High反转点（应该在局部高点）

---

## 下一步：PHASE2

现在有了高质量的反转点标记，可以进行：

1. **特征提取** - 从反转点周围提取K线特征
2. **模式识别** - 找出反转前的特征组合
3. **模型训练** - 训练分类器识别反转模式
4. **规则生成** - 从模型反向推导交易规则

---

## 总结

方向改变法：
- 逻辑清晰：只看方向改变
- 准确高效：标记实际的买卖点
- 易于理解：没有复杂指标
- 便于优化：参数调整灵活

这是从"技术指标"走向"纯价格行为"的重要一步。
