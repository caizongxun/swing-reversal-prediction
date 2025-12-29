# Swing Reversal Prediction Model - Crypto Trading

加密貨幣反轉點預測模型：從數學標記進階到智能過濾標記

## Project Vision

開發一個機器學習模型，精準檢測加密貨幣（BTCUSDT 15分鐘K線）的反轉點（Swing Reversals），用於識別可交易的底部（Low）和頂部（High）。

**核心問題**：傳統的本地極值（Local Min/Max）會產生大量的假訊號（Whipsaws），特別是在強勢趨勢中。我們需要建立一套智能過濾機制，只標記那些真正導致反向移動的反轉點。

## Project Roadmap

### Phase 1: Intelligent Labeling (Current)
- [x] 基於Window參數識別Local Peaks/Valleys
- [x] 實現智能過濾邏輯：只有當未來N根K線內的價格移動超過X%，才視為「真正反轉」
- [x] 生成乾淨的標籤數據集，區分真訊號與假訊號
- [x] 可視化分析結果

### Phase 2: Feature Engineering (Planned)
- [ ] Reversal特定特徵工程
  - RSI Divergence (divergence between price and RSI)
  - Candlestick Patterns (Hammer, Shooting Star, Engulfing)
  - Climax Volume (volume spike at reversal points)
  - Bollinger Bands %B (mean reversion indicator)
  - Volume Profile

### Phase 3: Model Training (Planned)
- [ ] Train LightGBM classifier
- [ ] True Reversal vs. Fake Signal classification
- [ ] Feature importance analysis
- [ ] Cross-validation and backtesting

### Phase 4: Deployment (Planned)
- [ ] Convert to trading signals
- [ ] Real-time prediction endpoint
- [ ] Backtesting framework

## Phase 1: Quick Start

### Installation

```bash
git clone https://github.com/caizongxun/swing-reversal-prediction.git
cd swing-reversal-prediction
pip install pandas numpy matplotlib
```

### Usage

#### Basic Usage

```bash
python phase1_analysis.py --data_path BTCUSDT_15m_binance_us.csv
```

#### Advanced Usage with Custom Parameters

```bash
python phase1_analysis.py \
  --data_path BTCUSDT_15m_binance_us.csv \
  --window 5 \
  --future_candles 12 \
  --threshold 1.0 \
  --output swing_reversal_analysis.png \
  --export_csv processed_btcusdt_15m.csv
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | Required | Path to BTCUSDT CSV file (OHLCV format) |
| `window` | 5 | Window size for swing point detection (look left/right) |
| `future_candles` | 12 | Number of candles ahead to check for reversal confirmation |
| `threshold` | 1.0 | Minimum price movement % required for confirmation |
| `output` | swing_reversal_analysis.png | Output visualization file |
| `export_csv` | None | Optional: export processed data to CSV |

### Example Output

The script generates:

1. **Visualization** (`swing_reversal_analysis.png`):
   - Top chart: Price candlesticks with swing points
     - Blue circles: Raw swing points (all local extrema)
     - Green triangles: Confirmed reversals (TRUE signals)
     - Red X marks: False signals (FILTERED out)
   - Bottom chart: Future price movement for each swing point
     - Green bars: Movements ≥ 1.0% (confirmed)
     - Red bars: Movements < 1.0% (not confirmed)

2. **Console Report**: Comprehensive statistics including:
   - Detection counts (raw vs. confirmed)
   - Confirmation rates
   - Filtering ratio
   - Future movement statistics

3. **Optional CSV Export**: Full dataset with all computed features

### Sample Output Report

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

## Technical Details

### Swing Point Detection Algorithm

For each candle at position `i`, we check if it's a swing point:

**Swing High**:
```
if high[i] == max(high[i-window:i+window]) and 
   high[i] > max(high[i-window:i-1]):
    → Mark as Swing High
```

**Swing Low**:
```
if low[i] == min(low[i-window:i+window]) and 
   low[i] < min(low[i-window:i-1]):
    → Mark as Swing Low
```

### Intelligent Filtering Logic

For each detected swing point, we validate by checking future price movement:

**For Swing High**:
```
if (close[i] - min(low[i+1:i+future_candles])) / close[i] * 100 >= threshold:
    → Mark as Confirmed Reversal (TRUE signal)
else:
    → Mark as False Signal (filtered out)
```

**For Swing Low**:
```
if (max(high[i+1:i+future_candles]) - close[i]) / close[i] * 100 >= threshold:
    → Mark as Confirmed Reversal (TRUE signal)
else:
    → Mark as False Signal (filtered out)
```

## File Structure

```
swing-reversal-prediction/
├── swing_reversal_detector.py      # Core detection module
├── phase1_analysis.py              # Main analysis script
├── README.md                       # This file
└── LICENSE
```

## Data Requirements

CSV file must contain OHLCV data with columns (case-insensitive):
- `open_time` or `timestamp`: Candle open time
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume (optional)

Example:
```
open_time,open,high,low,close,volume
2025-09-14 09:30:00,45000.00,45250.00,44950.00,45100.00,1523456
2025-09-14 09:45:00,45100.00,45400.00,45050.00,45350.00,1234567
```

## Next Steps (Phase 2)

Once Phase 1 labels are validated:

1. Add reversal-specific features
2. Explore feature correlations
3. Train classification model
4. Evaluate model performance
5. Deploy to production

## Notes

- **Window Parameter**: Affects sensitivity. Larger window = fewer but stronger swing points
- **Future Candles**: Should match your trading timeframe (e.g., 12x15m = 3 hours lookahead)
- **Threshold**: Balance between capturing opportunities and reducing false signals
- **Data Quality**: Ensure no gaps or missing data in CSV

## License

MIT

## Author

Swing Reversal Prediction Team
