# Phase 1 Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/caizongxun/swing-reversal-prediction.git
cd swing-reversal-prediction

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage (30 seconds)

Assuming you have `BTCUSDT_15m_binance_us.csv` in your working directory:

```bash
python phase1_analysis.py --data_path BTCUSDT_15m_binance_us.csv
```

This generates:
1. `swing_reversal_analysis.png` - Visualization
2. Console output with statistics

## Advanced Usage

### Option 1: Export Labeled Data

```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --export_csv labeled_btcusdt_15m.csv
```

Output: `labeled_btcusdt_15m.csv` with columns:
- timestamp, open, high, low, close, volume
- swing_type (high/low/None)
- raw_label (1=swing, 0=not)
- confirmed_label (1=true reversal, 0=false signal)
- future_move_pct (actual movement %)

### Option 2: Adjust Parameters

#### More Sensitive (Detect More Reversals)
```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --window 3 \
  --threshold 0.8 \
  --output sensitive_analysis.png
```

#### More Strict (Reduce False Signals)
```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --window 7 \
  --threshold 1.5 \
  --output strict_analysis.png
```

#### Shorter Confirmation Window (Scalping)
```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --future_candles 6 \
  --output scalp_analysis.png
```

#### Longer Confirmation Window (Swing Trading)
```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --future_candles 20 \
  --output swing_analysis.png
```

### Option 3: Full Analysis Pipeline

```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --window 5 \
  --future_candles 12 \
  --threshold 1.0 \
  --output phase1_results.png \
  --export_csv phase1_labeled_data.csv
```

## Understanding the Output

### Console Report Example

```
======================================================================
SWING REVERSAL DETECTION - PHASE 1 ANALYSIS REPORT
======================================================================

DETECTOR PARAMETERS:
  Window Size: 5
  Future Candles: 12
  Move Threshold: 1.0%

DATASET STATISTICS:
  Total Candles: 10,000
  Date Range: 2025-09-14 09:30:00 to 2025-12-27 13:15:00
  Price Range: $42,200.00 - $108,500.00

SWING DETECTION RESULTS:
  Total Raw Swing Points: 289
  ├─ Swing Highs Detected: 144
  └─ Swing Lows Detected: 145

INTELLIGENT FILTERING RESULTS:
  Confirmed True Reversals: 87
  ├─ Confirmed Highs: 42
  └─ Confirmed Lows: 45
  False Signals Filtered Out: 202
  Filtering Ratio: 30.10%
  (Only 30.10% of raw swings are TRUE reversals)

CONFIRMATION RATES:
  Swing High Confirmation Rate: 29.17%
  Swing Low Confirmation Rate: 31.03%

CONFIRMED REVERSALS STATISTICS:
  Average Future Movement: 1.45%
  Max Future Movement: 5.23%
  Min Future Movement: 1.00%

FALSE SIGNALS STATISTICS:
  Average Future Movement: 0.32%
  Max Future Movement: 0.99%
  Min Future Movement: 0.01%
======================================================================
```

**What This Means**:
- **289 raw swings** = All local extrema detected
- **87 confirmed** = Only 30% are true reversals (70% filtered noise)
- **1.45% avg move** for confirmed > **1.0% threshold** (good separation)
- **0.32% avg move** for false signals (clearly separated)

### Visualization Interpretation

**Top Chart**:
- **Black line** = Price (close)
- **Blue circles** = Raw swing highs (not yet confirmed)
- **Light blue circles** = Raw swing lows (not yet confirmed)
- **Green triangles pointing down** = Confirmed swing highs (TRUE reversals)
- **Green triangles pointing up** = Confirmed swing lows (TRUE reversals)
- **Red X marks** = False signals (filtered out)

**What to Look For**:
- Green triangles should appear at actual market reversals
- Red X marks should be in the middle of trends (fake signals)
- No green triangles in the middle of strong moves (good filtering)

**Bottom Chart**:
- **X-axis** = Candle position
- **Y-axis** = Future price movement % (in next 12 candles)
- **Green bars** = > 1% movement (confirmed)
- **Red bars** = < 1% movement (false signal)
- **Dashed line** = 1% threshold

**What to Look For**:
- Clear separation between green and red
- All green bars significantly above 1%
- All red bars significantly below 1%
- Few bars near the threshold (good discrimination)

## Performance Expectations

### Good Results

- Confirmed reversals: 20-40% of raw swings
- Avg move (confirmed): > 1.5%
- Avg move (false): < 0.5%
- High/Low confirmation rates: Within ±5%
- Separation gap: > 3x (1.5% vs 0.5%)

### Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| Too many false signals | Green and red bars overlapping, Confirmed > 50% | Increase threshold (1.0% → 1.5%), Increase window |
| Too few signals | Mostly red bars, Confirmed < 15% | Decrease threshold (1.0% → 0.8%), Decrease window |
| Unbalanced high/low | High 50%, Low 10% | Market trending, consider regime-specific models |
| Poor separation | Green bars barely above 1% | Increase threshold to 1.5% or 2.0% |

## Next Steps

### After Validating Phase 1

1. **Verify Results**
   - Check visualization against your manual market analysis
   - Confirm green triangles align with real reversals
   - Confirm red X's are indeed false signals

2. **Adjust Parameters if Needed**
   - Too many false signals? Increase threshold to 1.5%
   - Missing reversals? Decrease window to 3
   - Not enough data? Increase future_candles to 20

3. **Export Labeled Data**
   - Use `--export_csv` to get ground truth
   - This CSV becomes training data for Phase 2

4. **Prepare for Phase 2**
   - Phase 2 will add features (RSI, patterns, volume)
   - Each candle before a reversal gets feature vector
   - Train classifier on these features

## Python API (Advanced)

```python
from swing_reversal_detector import SwingReversalDetector
import pandas as pd

# Load data
df = pd.read_csv('BTCUSDT_15m_binance_us.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Initialize detector
detector = SwingReversalDetector(
    window=5,
    future_candles=12,
    move_threshold=1.0
)

# Run pipeline
results_df, stats = detector.process(df)

# Access results
print(f"Total raw swings: {stats['total_raw_swings']}")
print(f"Confirmed reversals: {stats['total_confirmed_reversals']}")

# Get confirmed reversals only
confirmed = results_df[results_df['confirmed_label'] == 1]
print(f"Confirmed reversal points:\n{confirmed[['timestamp', 'close', 'swing_type']]}")

# Export
results_df.to_csv('output.csv', index=False)
```

## FAQ

**Q: What if my CSV has different column names?**
A: The script auto-detects and handles common variations (open_time, timestamp, etc.).

**Q: Can I use this on hourly or daily data?**
A: Yes! Just adjust future_candles. For 1h: use 12 (12 hours). For daily: use 5-10 (weeks).

**Q: How do I know if my parameters are good?**
A: Look for 25-35% confirmation ratio and >3x separation in average moves.

**Q: Can I train a model on the output?**
A: Yes! Phase 2 will use this labeled data (confirmed_label) as ground truth.

**Q: What's the difference between raw_label and confirmed_label?**
A: raw_label = swing point detected (1=yes, 0=no). confirmed_label = validated with future movement (1=true, 0=false).

---

**Ready to Start?** Run this:
```bash
python phase1_analysis.py --data_path BTCUSDT_15m_binance_us.csv
```

**Questions?** Check PHASE1_TECHNICAL_GUIDE.md for detailed explanations.
