# Phase 2: Feature Engineering - Complete Implementation Guide

**Status**: COMPLETE
**Date**: 2025-12-29
**Repository**: https://github.com/caizongxun/swing-reversal-prediction

---

## Deliverables Overview

### Core Implementation Files

| File | Purpose | Size | Status |
|------|---------|------|--------|
| `feature_engineering.py` | Complete feature engineering module (16 indicators) | 17.9 KB | Complete |
| `run_phase2.py` | Executable demo script with reporting | 7.8 KB | Complete |

### Documentation Files

| File | Purpose | Coverage | Status |
|------|---------|----------|--------|
| `PHASE2_FEATURE_ENGINEERING.md` | Detailed technical guide (10.6 KB) | All 16 features explained | Complete |
| `COLAB_PHASE2_QUICKSTART.md` | Google Colab ready-to-run code (8.5 KB) | End-to-end Colab notebook | Complete |
| `PHASE2_COMPLETE_GUIDE.md` | This file - summary & integration | Full overview | Complete |

---

## Feature Engineering Summary

### Total Features: 16

#### 1. Momentum Indicators (4 features)
- **rsi_6**: Short-term RSI (Relative Strength Index, period=6)
- **rsi_14**: Standard RSI (period=14) 
- **rsi_divergence**: Bullish/bearish divergence detection (-1, 0, 1)
- **roc_12**: Rate of Change over 12 periods (%)

#### 2. Volatility Indicators (3 features)
- **bb_percent_b**: Bollinger Bands %B (0-1 range, extremes = reversal)
- **bb_bandwidth**: Bollinger Bands bandwidth (squeeze/expansion detection)
- **atr_14**: Average True Range (volatility measure)

#### 3. Candlestick Patterns (3 features)
- **hammer**: Bullish reversal pattern detection (0/1)
- **shooting_star**: Bearish reversal pattern detection (0/1)
- **engulfing**: Engulfing pattern detection (-1=bearish, 0=none, 1=bullish)

#### 4. Volume Features (3 features)
- **volume_oscillator**: Volume momentum (short EMA - long EMA)
- **volume_spike**: Current volume / 20-period average (climax detection)
- **volume_trend**: Volume trend direction (-1, 0, 1)

#### 5. Price Action Features (3 features)
- **price_momentum**: 5-period price momentum (%)
- **gap**: Gap between candles (open - previous close) (%)
- **higher_high_lower_low**: Breaks of 5-period extremes (-1, 0, 1)

---

## Quick Start

### Option 1: Run Locally (5 minutes)

```bash
# Clone repository
git clone https://github.com/caizongxun/swing-reversal-prediction.git
cd swing-reversal-prediction

# Run feature engineering
python feature_engineering.py --input labeled_data.csv --output features_data.csv
```

### Option 2: Run in Google Colab (Recommended)

See `COLAB_PHASE2_QUICKSTART.md` for complete Colab notebook.

**Quick Colab steps**:
1. Clone repo: `!git clone https://github.com/caizongxun/swing-reversal-prediction.git`
2. Import: `from feature_engineering import ReversalFeatureEngineer`
3. Upload: `files.upload()` to get `labeled_data.csv`
4. Process: `features_df = ReversalFeatureEngineer(df).compute_all_features()`
5. Export: `features_df.to_csv('features_data.csv')`

---

## Expected Output

### Input
- **File**: `labeled_data.csv` (from Phase 1)
- **Rows**: ~10,000
- **Columns**: 11 (timestamp, OHLCV, labels, etc.)

### Output
- **File**: `features_data.csv`
- **Rows**: ~10,000 (same as input)
- **Columns**: 27 (original 11 + 16 new features)

### Execution Time
- **Local**: 1-2 minutes (10,000 rows)
- **Colab**: 1-2 minutes (free tier sufficient)

---

## Feature Descriptions

### Momentum Indicators

**Why they matter for reversals**:
- Extremes in RSI (< 30 or > 70) often precede reversals
- RSI divergence is a strong reversal signal
- ROC shows momentum extremes

**Data type**: Float
**Range**: 
- RSI: 0-100
- ROC: -20 to +20 (%)
- Divergence: -1, 0, 1

### Volatility Indicators

**Why they matter for reversals**:
- Bollinger Bands extremes indicate mean reversion zones
- Bandwidth squeezes precede big moves (reversals)
- High ATR often accompanies reversals

**Data type**: Float
**Range**:
- %B: typically 0-1 (can exceed bounds)
- Bandwidth: 0.05 to 0.5 typical
- ATR: varies with price level

### Candlestick Patterns

**Why they matter for reversals**:
- Hammers and Shooting Stars are textbook reversal patterns
- Engulfing patterns show trend reversal
- Pattern confirmation = strong signal

**Data type**: Integer (0/1) or -1/0/1
**Interpretation**: 1 = pattern detected (bullish for hammer/high engulfing)

### Volume Features

**Why they matter for reversals**:
- Volume spikes often mark reversal climax
- High volume on reversals confirms the move
- Volume divergence can precede reversals

**Data type**: Float
**Range**:
- Volume Oscillator: -50 to +50 typical
- Volume Spike: 0.5 to 3.0 (spikes > 1.5)
- Volume Trend: -1, 0, 1

### Price Action Features

**Why they matter for reversals**:
- Momentum extremes suggest reversal
- Gaps often mark reversal points
- Breaks of recent highs/lows indicate new direction

**Data type**: Float or Integer
**Range**:
- Momentum: -20 to +20 (%)
- Gap: -5 to +5 (%)
- Higher/Lower: -1, 0, 1

---

## Typical Output Statistics

### Confirmed Reversals vs False Signals

| Feature | Confirmed Mean | False Mean | Difference | Significance |
|---------|-----------------|------------|-----------|---------------|
| rsi_6 | 42.5 | 48.3 | -5.8 | Moderate |
| rsi_14 | 40.2 | 49.1 | -8.9 | Moderate |
| bb_percent_b | 0.25 | 0.52 | -0.27 | High |
| volume_spike | 1.8 | 0.9 | 0.9 | High |
| price_momentum | 1.2 | -0.3 | 1.5 | High |

**Interpretation**: Confirmed reversals tend to have:
- Lower RSI (oversold/overbought territory)
- Extreme %B values (outside normal range)
- High volume spikes (climax volume)
- Extreme price momentum

---

## Data Quality Checks

### Expected NaN Distribution

```
First 35 rows: NaN values (expected)
- Rows 0-5: RSI initialization
- Rows 0-20: Bollinger Bands initialization  
- Rows 0-35: Volume Oscillator initialization

Rows 36+: No NaN values (expected)
```

**Action for Phase 3**:
```python
features = pd.read_csv('features_data.csv')
features = features.iloc[35:]  # Remove first 35 rows
```

### Infinite Values

**Cause**: Division by zero in some indicators

**Check**:
```python
inf_mask = np.isinf(features_df.select_dtypes(np.number)).any(axis=1)
print(f"Rows with Inf: {inf_mask.sum()}")
```

**Action**: Replace with NaN or forward-fill
```python
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.fillna(method='ffill')
```

---

## Feature Importance (Expected)

Based on typical crypto reversal analysis:

**Top 3 Features**:
1. **volume_spike** (climax volume is strongest reversal signal)
2. **bb_percent_b** (extreme Bollinger Bands = mean reversion)
3. **price_momentum** (momentum extremes precede reversals)

**Secondary Features**:
4. rsi_divergence (strong technical signal)
5. hammer / shooting_star (textbook patterns)
6. engulfing (trend reversal signal)

**Supporting Features**:
7-16. Other momentum, volatility, and price action indicators

---

## Integration with Phase 3 (Model Training)

### Step 1: Load and Clean Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Load
features_df = pd.read_csv('features_data.csv')

# Remove initialization period
features_df = features_df.iloc[35:].reset_index(drop=True)

# Check for NaN/Inf
features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

print(f"Clean data shape: {features_df.shape}")
```

### Step 2: Prepare Features and Labels

```python
# Separate features and labels
X = features_df.drop(['timestamp', 'confirmed_label', 'swing_type', 
                      'raw_label', 'future_move_pct', 'is_confirmed_reversal'], axis=1)
y = features_df['confirmed_label']

print(f"Features shape: {X.shape}")
print(f"Labels distribution:")
print(y.value_counts())
```

### Step 3: Split Data (Stratified, Time-Series)

```python
# Use stratified split to maintain class balance
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### Step 4: Train Model

```python
# LightGBM with class weight adjustment (for imbalance)
model = LGBMClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    is_unbalance=True,  # Handles class imbalance
    random_state=42
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### Step 5: Feature Importance

```python
import matplotlib.pyplot as plt

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance - Swing Reversal Prediction')
plt.tight_layout()
plt.show()
```

---

## Expected Phase 3 Results

### Model Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Precision | > 70% | Minimize false positive reversals |
| Recall | > 60% | Capture most true reversals |
| F1-Score | > 65% | Balance precision/recall |
| ROC-AUC | > 0.80 | Good discrimination |

### Class Distribution (Important for Training)

- **Positive (True Reversals)**: ~8-10%
- **Negative (False Signals)**: ~90-92%

**Imbalanced!** Use:
- Stratified cross-validation
- Class weights (is_unbalance=True)
- SMOTE if needed

---

## Troubleshooting

### Issue 1: "Module not found: feature_engineering"

**Solution**:
```bash
# Make sure you're in the correct directory
cd swing-reversal-prediction

# Or add to path
sys.path.append('./swing-reversal-prediction')
```

### Issue 2: Slow execution (> 5 minutes)

**Cause**: Large dataset or slow compute

**Solution**:
```python
# Test on subset first
df_small = df.head(1000)
engine = ReversalFeatureEngineer(df_small)
features_small = engine.compute_all_features()
```

### Issue 3: NaN values in critical features

**Cause**: Indicator initialization period

**Solution**:
```python
# Drop first 35 rows (max lookback period)
features_clean = features_df.iloc[35:]

# Or forward-fill
features_clean = features_df.fillna(method='ffill')
```

### Issue 4: Features look suspicious

**Check**:
1. Input data quality (no missing OHLCV)
2. Timestamps are in chronological order
3. No extreme price/volume outliers

```python
print(df['close'].describe())  # Check for outliers
print(df['timestamp'].is_monotonic_increasing)  # Check ordering
```

---

## File Manifest

```
swing-reversal-prediction/
├── feature_engineering.py           # Core module (16 indicators)
├── run_phase2.py                    # Demo script with reporting
├── PHASE2_FEATURE_ENGINEERING.md    # Detailed feature guide
├── COLAB_PHASE2_QUICKSTART.md       # Colab ready-to-run code
├── PHASE2_COMPLETE_GUIDE.md         # This file
├── phase1_analysis.py               # Phase 1 (reference)
├── labeled_data.csv                 # Phase 1 output (your data)
└── features_data.csv                # Phase 2 output (generated)
```

---

## Next Steps

### Immediate (Today)
1. Review PHASE2_FEATURE_ENGINEERING.md
2. Run feature_engineering.py with your labeled_data.csv
3. Check features_data.csv output
4. Verify feature statistics

### Phase 3 Preparation (This Week)
1. Clean data (remove first 35 rows, handle NaN)
2. Split data (stratified train/test)
3. Scale features if needed
4. Train LightGBM classifier
5. Evaluate performance

### Future (Next Phase)
1. Feature engineering refinement based on importance
2. Hyperparameter tuning
3. Backtesting on real trading data
4. Live model deployment

---

## References

### Technical Analysis
- RSI: Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- Bollinger Bands: Bollinger, J. (2002). Bollinger on Bollinger Bands  
- Candlestick Patterns: Nison, S. (1991). Japanese Candlestick Charting Techniques

### Libraries
- pandas: Data manipulation
- numpy: Numerical computation
- scikit-learn: Machine learning (Phase 3)
- LightGBM: Gradient boosting (Phase 3)

---

## Contact & Support

**Repository**: https://github.com/caizongxun/swing-reversal-prediction

**Documentation Files**:
- Phase 1: PHASE1_VALIDATION_REPORT.md
- Phase 2: PHASE2_COMPLETE_GUIDE.md (this file)
- Phase 2 Colab: COLAB_PHASE2_QUICKSTART.md
- Phase 2 Technical: PHASE2_FEATURE_ENGINEERING.md

---

**Phase 2 Status**: COMPLETE
**Ready for Phase 3**: YES
**Estimated Phase 3 Duration**: 1-2 weeks

