# Phase 1: Intelligent Swing Labeling - Technical Guide

## Overview

Phase 1 addresses a critical problem in swing reversal detection: **False Signals in Trends**.

When price is in a strong uptrend, the market creates many local highs and lows as it retraces. Traditional local extrema detection labels all of these as "swing points," but most are not true reversals—they're just retracements within the trend.

### The Problem

```
Strong Uptrend with False Signals:
        Peak A (FALSE)    Peak B (FALSE)    Peak C (REAL)
           /\                /\                /\
          /  \              /  \              /  \ ???
         /    \            /    \            /    \
--------/------\----------/------\----------/-------
       Low1    Low2      Low3    Low4      Low5

Problem: All peaks (A, B, C) are labeled as "swings"
But only Peak C leads to a real reversal (sustained downtrend)
```

### The Solution: Intelligent Filtering

We validate each detected swing point by checking if it's followed by significant price movement **in the expected direction**:

- **Swing High** = Local max followed by **drop > 1%** → TRUE reversal
- **Swing Low** = Local min followed by **rise > 1%** → TRUE reversal

This filters out trend retracements and preserves only actionable reversals.

## Mathematical Formulation

### Step 1: Swing Point Detection

For each candle `i` in the range `[window, len(data) - window]`:

**Swing High Detection**:
```
swing_high[i] = (
    high[i] == max(high[i-window : i+window]) AND
    high[i] > max(high[i-window : i-1])
)
```

Interpretation:
- Current high is the maximum in the entire window
- Current high is higher than all previous values (ensures we see increasing then decreasing)

**Swing Low Detection**:
```
swing_low[i] = (
    low[i] == min(low[i-window : i+window]) AND
    low[i] < min(low[i-window : i-1])
)
```

### Step 2: Intelligent Filtering (Future Movement Validation)

For each detected swing point at position `i`, validate it:

**For Swing High**:
```
downward_move % = ((close[i] - min(low[i+1 : i+future_candles])) / close[i]) * 100

is_confirmed = (downward_move % >= threshold %)
```

Interpretation:
- Calculate the maximum drop from the swing high to the lowest point in the next N candles
- If drop exceeds threshold (e.g., 1%), mark as TRUE reversal
- Otherwise, it's a FALSE signal (trend retracement) and filter it out

**For Swing Low**:
```
upward_move % = ((max(high[i+1 : i+future_candles]) - close[i]) / close[i]) * 100

is_confirmed = (upward_move % >= threshold %)
```

Interpretation:
- Calculate the maximum rise from the swing low to the highest point in the next N candles
- If rise exceeds threshold, mark as TRUE reversal
- Otherwise, filter out as false signal

## Algorithm Walkthrough

### Example: Processing BTCUSDT 15-min Data

**Parameters**:
- Window = 5 (look 5 candles left and right)
- Future Candles = 12 (check next 3 hours for 15-min timeframe)
- Threshold = 1.0% (minimum confirming movement)

**Dataset**: 10,000 candles from 2025-09-14 to 2025-12-27

**Typical Results**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Raw Swings | 289 | All local extrema detected |
| Confirmed Reversals | 87 | Only true reversals (30% of raw) |
| False Signals Filtered | 202 | Trend retracements removed (70%) |
| High Confirmation Rate | 29% | About 1 in 3 swing highs lead to reversal |
| Low Confirmation Rate | 31% | About 1 in 3 swing lows lead to reversal |

**Signal Quality**:
- Confirmed reversals: avg move **1.45%** (strong signals)
- False signals: avg move **0.32%** (weak, filtered out)

## Configuration Guide

### Window Size

**Effect**: Controls sensitivity of swing detection

| Window | Characteristics |
|--------|------------------|
| 3 | Very sensitive, detects micro-reversals, many false signals |
| **5** | **Balanced (RECOMMENDED)** |
| 7-8 | Less sensitive, captures only major reversals, fewer signals |
| 10+ | Very coarse, might miss tradeable opportunities |

**Default**: 5 (captures daily chart-like reversals on 15-min timeframe)

### Future Candles

**Effect**: Defines "look-ahead" period for confirming reversals

| Value | Timeframe (15m) | Use Case |
|-------|-----------------|----------|
| 4 | 1 hour | Scalping, very short-term |
| **12** | **3 hours (RECOMMENDED)** | Day trading |
| 20 | 5 hours | Medium-term swings |
| 96 | 24 hours | Daily reversals |

**Default**: 12 (3-hour confirmation window)

**Reasoning**: 3 hours is reasonable for swing reversals on 15-min charts—enough time for a meaningful move but not so long that you're looking at next-day data.

### Threshold

**Effect**: Minimum required price movement to confirm reversal

| Threshold | Characteristics |
|-----------|------------------|
| 0.5% | Very sensitive, more signals, lower quality |
| **1.0%** | **Balanced (RECOMMENDED)** |
| 1.5% | Higher bar, fewer signals, higher quality |
| 2.0%+ | Very restrictive, only strongest reversals |

**Default**: 1.0%

**Rationale**: In crypto, 1% move is noticeable and tradeable with reasonable risk/reward.

## Interpreting Results

### Visualization Breakdown

**Top Chart (Price & Swing Points)**:

1. **Black line**: Price (close)
2. **Blue circles**: Raw swing highs (not yet validated)
3. **Light blue circles**: Raw swing lows (not yet validated)
4. **Green triangles pointing down**: Confirmed swing highs (TRUE reversals)
5. **Green triangles pointing up**: Confirmed swing lows (TRUE reversals)
6. **Red X marks**: False signals (filtered out, did not confirm)

**Bottom Chart (Future Movement)**:

- **Green bars**: Swing points with >1% confirming movement (kept)
- **Red bars**: Swing points with <1% confirming movement (filtered)
- **Dashed line at 1.0%**: Confirmation threshold

### Statistics to Monitor

**Filtering Ratio**:
```
Filtering Ratio = Confirmed / Total Raw
```

- **Interpretation**: What percentage of raw swings are true reversals?
- **Typical**: 25-35% (good filtering, removing ~65-75% false signals)
- **Too high (>50%)**: Threshold might be too low, letting in false signals
- **Too low (<15%)**: Threshold might be too high, missing opportunities

**Confirmation Rate** (High vs. Low):
```
High Confirmation Rate = Confirmed Highs / Total Highs
Low Confirmation Rate = Confirmed Lows / Total Lows
```

- **Interpretation**: Are reversals equally likely at tops and bottoms?
- **Balanced**: Both around 25-35%
- **Imbalanced**: Might indicate directional bias in market data

**Future Movement Distribution**:

- **Confirmed Average**: Should be significantly higher than threshold (1.5-2%+)
- **False Signal Average**: Should be much lower (0.2-0.5%)
- **Large gap**: Good separation between signal quality

## Common Issues & Solutions

### Issue 1: Too Many False Signals (Confirmed < 20%)

**Symptoms**:
- Over 80% of swing points are filtered out
- Red X marks dominate the chart

**Solutions** (in order of recommendation):
1. Increase threshold (e.g., 1.0% → 1.5%)
2. Increase window (e.g., 5 → 7)
3. Increase future_candles (e.g., 12 → 20)

### Issue 2: Not Enough Signals (Confirmed > 50%)

**Symptoms**:
- Very few points filtered out
- Mostly green triangles, few red X marks

**Solutions**:
1. Decrease threshold (e.g., 1.0% → 0.8%)
2. Decrease window (e.g., 5 → 3)
3. Decrease future_candles (e.g., 12 → 8)

### Issue 3: Unequal High/Low Confirmation Rates

**Symptoms**:
- High Confirmation Rate ≈ 50%, Low Confirmation Rate ≈ 10%

**Interpretation**:
- Market is in strong uptrend (tops easier to reversal, bottoms harder)
- Adjust parameters based on market regime

**Solution**:
- Consider training separate models for different market conditions

## Output Interpretation

### CSV Export Columns

When exporting with `--export_csv`, you get:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Candle open time |
| open, high, low, close | float | OHLC prices |
| volume | float | Trading volume |
| swing_type | str | 'high' / 'low' / None |
| raw_label | int | 1 if swing point, 0 otherwise |
| confirmed_label | int | 1 if TRUE reversal, 0 otherwise |
| future_move_pct | float | Actual price movement in next 12 candles |
| is_confirmed_reversal | bool | Same as confirmed_label (boolean format) |

### Using Labeled Data

The output CSV is "ground truth" for Phase 2 (Feature Engineering):

```python
# Load labeled data
df = pd.read_csv('processed_btcusdt_15m.csv')

# Select confirmed reversals for feature engineering
true_reversals = df[df['confirmed_label'] == 1]

# Extract features from X candles before each reversal
feature_window = 10  # Use previous 10 candles
X_samples = []
y_labels = []

for idx in true_reversals.index:
    if idx >= feature_window:
        X_samples.append(df.iloc[idx - feature_window : idx])
        y_labels.append(true_reversals.loc[idx, 'confirmed_label'])

# Now compute reversal-specific features on X_samples
# RSI, candle patterns, volume, etc.
```

## Validation Checklist

Before proceeding to Phase 2, verify:

- [ ] Confirmed reversals show >1.5% average move (good separation from threshold)
- [ ] False signals show <0.5% average move (clear distinction)
- [ ] Filtering ratio is 20-40% (balanced)
- [ ] Confirmation rates for highs/lows are similar (±5%)
- [ ] Visual inspection: Green triangles appear at actual market reversals
- [ ] No data gaps in input CSV
- [ ] Timestamp column is properly formatted

## Next Steps: Phase 2 Preview

Once Phase 1 labels are validated, Phase 2 will:

1. **Extract features** from the 10 candles *before* each swing point:
   - RSI divergence (RSI higher but price lower = bullish divergence)
   - Candle patterns (Hammer, Shooting Star, Engulfing)
   - Volume climax (volume spike at reversal points)
   - Bollinger Bands %B (mean reversion setup)
   - MACD, Stochastic, ATR

2. **Create training dataset**:
   ```
   [candle_1_features, candle_2_features, ..., candle_10_features] → Label (True/False)
   ```

3. **Train classifier** (LightGBM):
   - Binary classification: True Reversal vs. False Signal
   - Cross-validation to prevent overfitting
   - Feature importance analysis

## References

- Swing Trading concepts: John Carter's "Mastering the Trade"
- Local extrema detection: Common technical analysis technique
- Filtering approach: Similar to confirmation mechanisms in Elliott Wave and Wyckoff analysis

---

**Last Updated**: 2025-12-29
**Phase**: 1/4 (Intelligent Labeling)
**Status**: Ready for Phase 2
