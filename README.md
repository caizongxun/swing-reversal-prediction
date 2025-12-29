# Swing Reversal Prediction System

A machine learning system for predicting swing reversals in cryptocurrency markets using technical indicators and candlestick patterns.

**Status**: Phase 2 Complete (Feature Engineering)

---

## Project Overview

This project develops a predictive model for identifying true swing reversals vs. false signals in crypto price charts. The approach uses intelligent labeling of historical data combined with machine learning.

### Key Features

- **Ground Truth Labeling (Phase 1)**: Intelligent filtering of reversal signals using volatility and confirmation criteria
- **Feature Engineering (Phase 2)**: 16 technical indicators across 5 categories
- **Model Training (Phase 3)**: LightGBM classifier with class imbalance handling

---

## Project Phases

### Phase 1: Intelligent Labeling COMPLETED

**Goal**: Create ground truth labels from OHLCV data

**Output**: `labeled_data.csv` with confirmed vs. false reversals

**Key Results**:
- Identified 87 confirmed reversals from 980 raw signals
- 8.9% true reversal rate (highly imbalanced dataset)
- Used Bollinger Bands and future price validation for confirmation

**Reference**: `phase1_analysis.py`, `labeled_data.csv`

---

### Phase 2: Feature Engineering COMPLETED ✓

**Goal**: Engineer 16 technical indicators to predict reversals

**Deliverables**:

#### Code Files
- **`feature_engineering.py`** (17.9 KB)
  - Core module with 16 indicator implementations
  - No external TA-lib dependency (custom, transparent code)
  - Full docstrings and examples
  - Classes: `ReversalFeatureEngineer`

- **`run_phase2.py`** (7.8 KB)
  - Executable demo script
  - Computes all features with reporting
  - Shows feature statistics and comparisons
  - Generates `features_data.csv` output

#### Documentation
- **`PHASE2_FEATURE_ENGINEERING.md`** - Technical deep-dive on all 16 indicators
- **`COLAB_PHASE2_QUICKSTART.md`** - Ready-to-run Google Colab notebook
- **`PHASE2_COMPLETE_GUIDE.md`** - Integration guide for Phase 3
- **`README.md`** - This file

#### Output Data
- **`features_data.csv`** - Input for Phase 3 model training
  - 980 rows (same as labeled_data.csv)
  - 27 columns (11 original + 16 new features)
  - Ready for machine learning

---

## Phase 2 Features

### 16 Technical Indicators (5 Categories)

#### 1. Momentum Indicators (4)
- `rsi_6`: Short-term RSI momentum
- `rsi_14`: Standard RSI
- `rsi_divergence`: Bullish/bearish divergence detection
- `roc_12`: Rate of change over 12 periods

#### 2. Volatility Indicators (3)
- `bb_percent_b`: Bollinger Bands %B (mean reversion indicator)
- `bb_bandwidth`: Bollinger Bands bandwidth (squeeze/expansion detector)
- `atr_14`: Average True Range (volatility measure)

#### 3. Candlestick Patterns (3)
- `hammer`: Bullish reversal pattern
- `shooting_star`: Bearish reversal pattern
- `engulfing`: Two-candle reversal pattern

#### 4. Volume Features (3)
- `volume_oscillator`: Short-term vs. long-term volume momentum
- `volume_spike`: Current volume relative to 20-period average
- `volume_trend`: Volume trend direction

#### 5. Price Action (3)
- `price_momentum`: 5-period price momentum
- `gap`: Gap between consecutive candles
- `higher_high_lower_low`: Breaks of 5-period extremes

---

## Quick Start

### Run Locally (5 minutes)

```bash
# Clone repository
git clone https://github.com/caizongxun/swing-reversal-prediction.git
cd swing-reversal-prediction

# Run feature engineering
python feature_engineering.py --input labeled_data.csv --output features_data.csv
```

### Run in Google Colab (Recommended)

See `COLAB_PHASE2_QUICKSTART.md` for complete notebook with all code.

**Summary**:
1. Clone repo in Colab
2. Upload `labeled_data.csv`
3. Run feature engineering
4. Download `features_data.csv`

**Estimated time**: 1-2 minutes

---

## Phase 2 Expected Output

### Input
```
labeled_data.csv (from Phase 1)
- 980 rows
- 11 columns (timestamp, OHLCV, labels)
```

### Output
```
features_data.csv
- 980 rows (same)
- 27 columns (original 11 + 16 features)
```

### Feature Statistics (Example)

**For Confirmed Reversals** (87 samples):
- RSI-14 Mean: 40.2 (oversold/overbought)
- Volume Spike Mean: 1.8x (climax volume)
- Price Momentum Mean: 1.2% (extreme moves)

**For False Signals** (893 samples):
- RSI-14 Mean: 49.1 (neutral)
- Volume Spike Mean: 0.9x (normal)
- Price Momentum Mean: -0.3% (random)

**Key Insight**: True reversals have extreme indicator values

---

## File Structure

```
swing-reversal-prediction/
├── README.md                           # This file
│
├── Phase 1: Labeling
├── phase1_analysis.py                  # Phase 1 labeling code
├── labeled_data.csv                    # Phase 1 ground truth
│
├── Phase 2: Feature Engineering (COMPLETE)
├── feature_engineering.py              # Core module (16 indicators)
├── run_phase2.py                       # Executable demo script
├── features_data.csv                   # Phase 2 output
│
├── Documentation
├── PHASE2_COMPLETE_GUIDE.md            # Integration guide
├── PHASE2_FEATURE_ENGINEERING.md       # Technical reference
├── COLAB_PHASE2_QUICKSTART.md          # Colab notebook
│
└── Phase 3: Model Training (Coming)
    ├── model_training.py               # LightGBM trainer
    ├── model_evaluation.py             # Evaluation metrics
    └── model.pkl                       # Trained model
```

---

## Next Steps: Phase 3

### Timeline
- **Start**: Week of Jan 5, 2026
- **Duration**: 1-2 weeks
- **Deliverables**: Trained model with >70% precision

### Phase 3 Plan

1. **Data Preparation**
   - Load `features_data.csv`
   - Remove initialization period (first 35 rows)
   - Handle NaN/Inf values
   - Feature scaling if needed

2. **Model Development**
   - Train LightGBM classifier
   - Handle class imbalance (90/10 split)
   - Stratified cross-validation
   - Hyperparameter tuning

3. **Evaluation**
   - Precision > 70% (minimize false reversals)
   - Recall > 60% (catch real reversals)
   - ROC-AUC > 0.80
   - Feature importance analysis

4. **Output**
   - Trained model file
   - Performance report
   - Feature importance rankings
   - Deployment guide

---

## Usage Examples

### Example 1: Run Feature Engineering on Your Data

```python
from feature_engineering import ReversalFeatureEngineer
import pandas as pd

# Load your labeled data
df = pd.read_csv('labeled_data.csv')

# Initialize feature engineer
engineering = ReversalFeatureEngineer(df)

# Compute all 16 features
features_df = engineer.compute_all_features()

# Export
features_df.to_csv('features_data.csv', index=False)
```

### Example 2: Check Feature Statistics

```python
features_df = pd.read_csv('features_data.csv')

# For confirmed reversals only
confirmed = features_df[features_df['confirmed_label'] == 1]
print(confirmed[['rsi_14', 'bb_percent_b', 'volume_spike']].describe())

# Compare with false signals
false_sig = features_df[features_df['confirmed_label'] == 0]
print(false_sig[['rsi_14', 'bb_percent_b', 'volume_spike']].describe())
```

### Example 3: Prepare for Model Training (Phase 3)

```python
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Load
df = pd.read_csv('features_data.csv')
df = df.iloc[35:]  # Remove NaN initialization

# Prepare
X = df.drop(['timestamp', 'confirmed_label'], axis=1)
y = df['confirmed_label']

# Split (stratified for class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train
model = LGBMClassifier(is_unbalance=True)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

print(f"Train score: {model.score(X_train, y_train):.4f}")
print(f"Test score: {model.score(X_test, y_test):.4f}")
```

---

## Key Insights

### Feature Engineering Results

1. **Momentum indicators** show strong correlation with reversals
   - RSI extremes (< 30 or > 70) are reliable signals
   - RSI divergence is highly predictive

2. **Volume spikes** are critical for confirmation
   - True reversals often have 1.5x+ normal volume
   - Without volume, patterns are less reliable

3. **Candlestick patterns** work in conjunction with indicators
   - Patterns alone have high false positive rate
   - Combined with volume/momentum: much stronger

4. **Bollinger Bands %B** captures mean reversion
   - Extremes (< -0.2 or > 1.2) indicate reversal zones
   - Used as primary entry/exit filter

---

## Data Quality

### Input Requirements
- OHLCV data in chronological order
- No missing values in price/volume
- At least 500 candles for indicator calculation
- 15-minute or higher timeframe recommended

### Output Quality
- First 35 rows have NaN (indicator initialization)
- No NaN or Inf in rows 36+
- Feature values in expected ranges
- All data types numeric (float/int)

---

## Dependencies

**Phase 2 Dependencies**:
- pandas >= 1.0
- numpy >= 1.18
- No external technical analysis library required

**Phase 3 Dependencies** (optional):
- scikit-learn >= 0.24
- lightgbm >= 3.0
- matplotlib (for visualization)

```bash
# Install
pip install pandas numpy scikit-learn lightgbm matplotlib
```

---

## Troubleshooting

### Issue: "Module not found: feature_engineering"
```bash
cd swing-reversal-prediction  # Make sure you're in right directory
python feature_engineering.py
```

### Issue: Slow execution
```python
# Test on small subset first
df_test = df.head(100)
engineering = ReversalFeatureEngineer(df_test)
features = engineer.compute_all_features()
```

### Issue: NaN values in output
```python
# Expected in first 35 rows (initialization period)
df = pd.read_csv('features_data.csv')
df = df.iloc[35:]  # Remove NaN
```

---

## References

### Technical Analysis Books
- Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- Bollinger, J. (2002). Bollinger on Bollinger Bands
- Nison, S. (1991). Japanese Candlestick Charting Techniques

### Academic Papers
- [RSI Documentation](https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index)
- [Bollinger Bands Guide](https://en.wikipedia.org/wiki/Bollinger_Bands)

---

## License

MIT License - See LICENSE file

---

## Contact

**Repository**: https://github.com/caizongxun/swing-reversal-prediction

**Issues & Discussions**: GitHub Issues

---

## Project Timeline

| Phase | Status | Completion | Deliverables |
|-------|--------|------------|--------------|
| 1. Labeling | COMPLETE | Dec 15, 2025 | labeled_data.csv |
| 2. Features | COMPLETE | Dec 29, 2025 | features_data.csv + 16 indicators |
| 3. Training | IN PROGRESS | Jan 15, 2026 | Trained model + metrics |
| 4. Deployment | PLANNED | Feb 2026 | Live trading system |

---

**Last Updated**: 2025-12-29
**Phase 2 Status**: READY FOR PHASE 3

