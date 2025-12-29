# Phase 2: Feature Engineering for Swing Reversal Classification

## Overview

Phase 2 transforms the ground truth labels from Phase 1 into a rich feature set that captures the technical characteristics of true reversals vs. false signals.

**Input**: `labeled_data.csv` (from Phase 1)
**Output**: `features_data.csv` (input for Phase 3 model training)

---

## Feature Categories

### 1. Momentum Indicators (4 features)

#### RSI (Relative Strength Index)

**Periods**: 6, 14

```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss
```

**For Reversals**:
- RSI < 30: Oversold (potential bullish reversal)
- RSI > 70: Overbought (potential bearish reversal)
- RSI at extremes often precedes reversals

**Columns**:
- `rsi_6`: Short-term momentum
- `rsi_14`: Standard momentum

#### RSI Divergence

**Detection Logic**:
- **Bullish Divergence**: Price makes lower low, but RSI makes higher low (bullish reversal signal)
- **Bearish Divergence**: Price makes higher high, but RSI makes lower high (bearish reversal signal)

**Column**: `rsi_divergence`
- Value: 1 (bullish), -1 (bearish), 0 (none)

**Significance**: Strong predictor of reversals; often confirms true reversal points

#### Rate of Change (ROC)

```
ROC = ((Close - Close[n periods ago]) / Close[n periods ago]) * 100
```

**Column**: `roc_12`
- Measures price momentum
- Extreme values (>10% or <-10%) suggest reversal potential

---

### 2. Volatility Indicators (3 features)

#### Bollinger Bands %B

```
%B = (Close - Lower Band) / (Upper Band - Lower Band)
```

**Interpretation**:
- %B < 0: Price below lower band (mean reversion zone, bullish reversal potential)
- %B > 1: Price above upper band (mean reversion zone, bearish reversal potential)
- %B = 0.5: Price at moving average (neutral)

**Column**: `bb_percent_b`
- Range: typically 0-1, can exceed bounds
- Extremes (< -0.2 or > 1.2) are strong reversal signals

**Key Insight**: True reversals often occur when %B is at extremes

#### Bollinger Bands Bandwidth

```
Bandwidth = (Upper Band - Lower Band) / Middle Band
```

**Interpretation**:
- **Low Bandwidth** (< 0.1): Squeeze condition
  - Markets consolidate before big moves
  - Precedes reversals
  - Low volatility compression

- **High Bandwidth** (> 0.3): Expansion condition
  - Volatility spike
  - Often accompanies reversals
  - Market break out of consolidation

**Column**: `bb_bandwidth`
- Used to detect squeeze-breakout scenarios
- Extremes (very low or very high) correlate with reversals

#### Average True Range (ATR)

```
TR = max(
    High - Low,
    |High - Previous Close|,
    |Low - Previous Close|
)
ATR = 14-period EMA of TR
```

**Column**: `atr_14`
- Measures volatility
- High ATR: High volatility environment (reversal-prone)
- Low ATR: Low volatility, less reversal likelihood

**For Reversals**: Reversals often occur with ATR expansion

---

### 3. Candlestick Patterns (3 features)

#### Hammer

**Pattern Characteristics**:
- Small body (open/close near each other)
- Long lower shadow (at least 2-3x body size)
- Short/no upper shadow
- Appears after downtrend
- **Signal**: Bullish reversal

**Column**: `hammer` (1 = detected, 0 = not detected)

**Criteria in Code**:
```
Lower Shadow > Body Size * 2
Upper Shadow < Body Size * 0.5
Lower Shadow > Total Range * 0.5
```

**Significance**: Strong bullish reversal signal when confirmed

#### Shooting Star

**Pattern Characteristics**:
- Small body (open/close near each other)
- Long upper shadow (at least 2-3x body size)
- Short/no lower shadow
- Appears after uptrend
- **Signal**: Bearish reversal

**Column**: `shooting_star` (1 = detected, 0 = not detected)

**Criteria in Code**:
```
Upper Shadow > Body Size * 2
Lower Shadow < Body Size * 0.5
Upper Shadow > Total Range * 0.5
```

**Significance**: Strong bearish reversal signal

#### Engulfing

**Pattern Characteristics**:
- Current candle completely engulfs previous candle
- Opposite direction from previous candle
- **Signal**: Trend reversal

**Column**: `engulfing` (1 = bullish, -1 = bearish, 0 = none)

**Criteria**:
- **Bullish Engulfing**:
  - Previous: Bearish (close < open)
  - Current: Bullish (close > open)
  - Current low < Previous low
  - Current high > Previous high

- **Bearish Engulfing**: Opposite conditions

**Significance**: Moderate to strong reversal signal

---

### 4. Volume Features (3 features)

#### Volume Oscillator

```
VO = (5-period EMA of Volume - 35-period EMA of Volume) / 35-period EMA * 100
```

**Column**: `volume_oscillator`

**Interpretation**:
- VO > 0: Short-term volume > long-term (buying/selling pressure increasing)
- VO < 0: Short-term volume < long-term (pressure decreasing)
- Extreme VO often confirms reversals

#### Volume Spike

```
Volume Spike = Current Volume / (20-period Average Volume)
```

**Column**: `volume_spike`

**Interpretation**:
- Value > 1.5: Volume spike (climax volume)
  - Often indicates reversal completion
  - High conviction move
  - Market capitulation

- Value < 0.7: Below average volume
  - Weak move, less reliable
  - May not hold

**Key**: True reversals often have volume spikes (> 1.5)

#### Volume Trend

**Logic**:
```
if Current Volume > 5-period SMA * 1.1:
    Trend = 1 (increasing)
elif Current Volume < 5-period SMA * 0.9:
    Trend = -1 (decreasing)
else:
    Trend = 0 (neutral)
```

**Column**: `volume_trend` (-1, 0, 1)

**For Reversals**: Increasing volume on reversal bars confirms move

---

### 5. Price Action Features (3 features)

#### Price Momentum

```
Momentum = ((Current Close - Close[5 periods ago]) / Close[5 periods ago]) * 100
```

**Column**: `price_momentum`

**Interpretation**:
- Positive: Upward momentum
- Negative: Downward momentum
- Extreme values indicate potential reversal

#### Gap Detection

```
Gap = ((Current Open - Previous Close) / Previous Close) * 100
```

**Column**: `gap`

**Interpretation**:
- Positive gap: Gap up (bullish)
- Negative gap: Gap down (bearish)
- Gaps often mark reversal points
- Large gaps: Strong reversal signals

#### Higher High / Lower Low

**Logic**:
```
if Current High > 5-period High:
    Pattern = 1 (higher high, bullish)
elif Current Low < 5-period Low:
    Pattern = -1 (lower low, bearish)
else:
    Pattern = 0 (neutral)
```

**Column**: `higher_high_lower_low` (-1, 0, 1)

**For Reversals**: Breaks of previous 5-period extremes often signal reversal

---

## Usage

### Quick Start

```bash
python feature_engineering.py --input labeled_data.csv --output features_data.csv
```

### Output Specification

**Columns in features_data.csv**:

**Original Columns**:
- timestamp, open, high, low, close, volume
- swing_type, raw_label, confirmed_label
- future_move_pct, is_confirmed_reversal

**New Feature Columns** (16 total):

**Momentum** (4):
- rsi_6, rsi_14, rsi_divergence, roc_12

**Volatility** (3):
- bb_percent_b, bb_bandwidth, atr_14

**Candlestick Patterns** (3):
- hammer, shooting_star, engulfing

**Volume** (3):
- volume_oscillator, volume_spike, volume_trend

**Price Action** (3):
- price_momentum, gap, higher_high_lower_low

### Sample Output

```
FEATURE ENGINEERING COMPLETE

Features created:
   1. rsi_6
   2. rsi_14
   3. rsi_divergence
   4. roc_12
   5. bb_percent_b
   6. bb_bandwidth
   7. atr_14
   8. hammer
   9. shooting_star
  10. engulfing
  11. volume_oscillator
  12. volume_spike
  13. volume_trend
  14. price_momentum
  15. gap
  16. higher_high_lower_low

Total features: 16
Total columns: 27 (original 11 + 16 features)

Confirmed reversals: 87 (8.9%)

Feature statistics for CONFIRMED reversals:
        rsi_6  rsi_14  rsi_divergence  roc_12  ...
mean   45.23   42.15              0.12     0.89  ...
std    18.34   16.21              0.32     2.34  ...
min     5.12   8.45              -1.00    -8.50  ...
max    95.67   92.34               1.00    12.30  ...
```

---

## Feature Analysis for Model Training

### Class Distribution

Typical from Phase 1 output:
- **Confirmed Reversals** (Label = 1): ~8-10%
- **False Signals** (Label = 0): ~90-92%

**Note**: Imbalanced dataset. Use stratified cross-validation in Phase 3.

### Feature Correlations

**Expected High Correlations**:
- rsi_6 and rsi_14 (similar indicators, different periods)
- volume_spike and volume_oscillator (both volume-based)
- price_momentum and roc_12 (both momentum)

**Expected Low Correlations**:
- Candlestick patterns and RSI (different signal types)
- Gaps and ATR (different aspects of price action)

### Data Quality

**NaN Handling**:
- First 20 rows will have NaN (SMA/EMA initialization)
- RSI needs at least period+1 rows
- Solution: Drop first 35 rows or forward-fill

**Recommendation** for Phase 3:
```python
# Drop first 35 rows (max lookback period)
X = features_data.iloc[35:]
y = X['confirmed_label']
X = X.drop(['confirmed_label', 'timestamp'], axis=1)
```

---

## Next Steps: Phase 3

### Input for Model Training

1. Load `features_data.csv`
2. Remove first 35 rows (NaN period)
3. Split features (X) and label (y)
4. Use stratified train/test split (due to class imbalance)

### Recommended Model

**LightGBM Classifier**
- Handles imbalanced data well
- Fast training
- Good interpretability (feature importance)
- Robust to outliers

### Expected Results

**Phase 3 Targets**:
- **Precision**: > 70% (minimize false positives)
- **Recall**: > 60% (capture most true reversals)
- **F1-Score**: > 65%
- **ROC-AUC**: > 0.80

---

## Troubleshooting

### Issue: NaN values in output

**Cause**: Indicator initialization period

**Solution**:
```python
# Fill NaN with forward fill
features_data = features_data.fillna(method='bfill')
features_data = features_data.fillna(method='ffill')

# Or drop first N rows
features_data = features_data.iloc[35:]
```

### Issue: Feature values seem wrong

**Check**:
1. Input CSV has correct OHLCV columns
2. Timestamps are sorted chronologically
3. No extreme outliers in price/volume

### Issue: Slow execution

**Optimization**:
- Use vectorized operations where possible
- Reduce lookback periods for testing
- Run on smaller data subset first

---

## References

### Technical Analysis
- RSI: Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- Bollinger Bands: Bollinger, J. (2002). Bollinger on Bollinger Bands
- Candlestick Patterns: Nison, S. (1991). Japanese Candlestick Charting Techniques

### Libraries Used
- pandas: Data manipulation
- numpy: Numerical computation
- No external TA library (custom implementations for transparency)

---

**Phase 2 Status**: READY
**Estimated Phase 3 Duration**: 1-2 weeks (model training + validation)

