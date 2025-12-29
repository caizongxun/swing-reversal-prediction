"""
Phase 2 Execution Script - Feature Engineering Demo

This script demonstrates Phase 2 in action with sample data.
You can run it locally or in Colab.

Usage:
    python run_phase2.py
"""

import pandas as pd
import numpy as np
from feature_engineering import ReversalFeatureEngineer
import os

def main():
    print("="*80)
    print("PHASE 2: FEATURE ENGINEERING FOR SWING REVERSAL PREDICTION")
    print("="*80)
    
    # Step 1: Check if labeled_data.csv exists
    if not os.path.exists('labeled_data.csv'):
        print("\nERROR: labeled_data.csv not found!")
        print("Please ensure Phase 1 output is available.")
        print("\nRunning with sample data instead...\n")
        df = create_sample_data()
    else:
        print("\nLoading labeled_data.csv...")
        df = pd.read_csv('labeled_data.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Confirmed reversals: {(df['confirmed_label'] == 1).sum()}")
    
    # Step 2: Initialize feature engineer
    print("\n" + "-"*80)
    print("INITIALIZING FEATURE ENGINEER")
    print("-"*80)
    engineer = ReversalFeatureEngineer(df)
    
    # Step 3: Compute all features
    print("\n" + "-"*80)
    print("COMPUTING FEATURES")
    print("-"*80)
    features_df = engineer.compute_all_features()
    
    # Step 4: Display results
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    
    # List all features
    original_cols = set(df.columns)
    new_cols = [col for col in features_df.columns if col not in original_cols]
    
    print(f"\nNew features created: {len(new_cols)}")
    print("\nFeature breakdown:")
    print("  Momentum (4): rsi_6, rsi_14, rsi_divergence, roc_12")
    print("  Volatility (3): bb_percent_b, bb_bandwidth, atr_14")
    print("  Patterns (3): hammer, shooting_star, engulfing")
    print("  Volume (3): volume_oscillator, volume_spike, volume_trend")
    print("  Price Action (3): price_momentum, gap, higher_high_lower_low")
    
    # Show head of data with selected features
    print("\n" + "-"*80)
    print("SAMPLE DATA (First 20 rows with selected features)")
    print("-"*80)
    display_cols = ['timestamp', 'close', 'volume', 'confirmed_label',
                   'rsi_14', 'bb_percent_b', 'hammer', 'volume_spike', 'price_momentum']
    print(features_df[display_cols].head(20).to_string())
    
    # Feature statistics for confirmed reversals
    print("\n" + "-"*80)
    print("FEATURE STATISTICS FOR CONFIRMED REVERSALS (confirmed_label=1)")
    print("-"*80)
    
    confirmed = features_df[features_df['confirmed_label'] == 1]
    if len(confirmed) > 0:
        stats_cols = ['rsi_6', 'rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum']
        print(f"\nTotal confirmed reversals: {len(confirmed)}")
        print(f"\nDescriptive statistics for key features:")
        print(confirmed[stats_cols].describe().to_string())
    else:
        print("\nNo confirmed reversals in sample data.")
    
    # Feature statistics for false signals
    print("\n" + "-"*80)
    print("FEATURE STATISTICS FOR FALSE SIGNALS (confirmed_label=0)")
    print("-"*80)
    
    false_signals = features_df[features_df['confirmed_label'] == 0]
    if len(false_signals) > 0:
        stats_cols = ['rsi_6', 'rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum']
        print(f"\nTotal false signals: {len(false_signals)}")
        print(f"\nDescriptive statistics for key features:")
        print(false_signals[stats_cols].describe().to_string())
    
    # Compare means
    print("\n" + "-"*80)
    print("MEAN FEATURE VALUES COMPARISON")
    print("-"*80)
    
    if len(confirmed) > 0 and len(false_signals) > 0:
        comparison_cols = ['rsi_6', 'rsi_14', 'bb_percent_b', 'volume_spike', 'price_momentum', 'atr_14']
        print(f"\n{'Feature':<20} {'Confirmed Mean':>15} {'False Mean':>15} {'Difference':>15}")
        print("-" * 65)
        
        for col in comparison_cols:
            conf_mean = confirmed[col].mean()
            false_mean = false_signals[col].mean()
            diff = conf_mean - false_mean
            print(f"{col:<20} {conf_mean:>15.4f} {false_mean:>15.4f} {diff:>15.4f}")
    
    # Feature correlations with label
    print("\n" + "-"*80)
    print("FEATURE CORRELATION WITH LABEL (confirmed_label)")
    print("-"*80)
    
    corr_cols = new_cols
    correlations = features_df[corr_cols + ['confirmed_label']].corr()['confirmed_label'].drop('confirmed_label')
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    
    print(f"\n{'Feature':<25} {'Correlation':>15}")
    print("-" * 40)
    for feat, corr in correlations_sorted.head(10).items():
        print(f"{feat:<25} {correlations[feat]:>15.4f}")
    
    # Data quality check
    print("\n" + "-"*80)
    print("DATA QUALITY CHECK")
    print("-"*80)
    
    nan_count = features_df[new_cols].isna().sum()
    print(f"\nTotal rows: {len(features_df)}")
    print(f"Features with NaN: {nan_count.sum()}")
    if nan_count.sum() > 0:
        print("\nNaN distribution:")
        for col, count in nan_count[nan_count > 0].items():
            print(f"  {col}: {count} rows")
    
    # Inf check
    inf_count = np.isinf(features_df[new_cols].select_dtypes(np.number)).sum().sum()
    print(f"Features with Inf: {inf_count}")
    
    # Export results
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80)
    
    output_file = 'features_data.csv'
    features_df.to_csv(output_file, index=False)
    print(f"\nExport complete!")
    print(f"File: {output_file}")
    print(f"Size: {len(features_df)} rows, {len(features_df.columns)} columns")
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    
    print(f"\nInput file: labeled_data.csv")
    print(f"Output file: {output_file}")
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"New features: {len(new_cols)}")
    print(f"Total columns in output: {len(features_df.columns)}")
    print(f"\nGround truth balance:")
    print(f"  Confirmed Reversals (1): {(features_df['confirmed_label'] == 1).sum()} ({(features_df['confirmed_label'] == 1).sum()/len(features_df)*100:.1f}%)")
    print(f"  False Signals (0): {(features_df['confirmed_label'] == 0).sum()} ({(features_df['confirmed_label'] == 0).sum()/len(features_df)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Load features_data.csv")
    print("  2. Drop first 35 rows (NaN initialization period)")
    print("  3. Proceed to Phase 3: Model Training (LightGBM)")
    print("="*80)
    
    return features_df

def create_sample_data(n_rows=1000):
    """
    Create synthetic sample data for testing.
    """
    print("Creating synthetic sample data...")
    
    np.random.seed(42)
    dates = pd.date_range('2025-09-14', periods=n_rows, freq='15min')
    
    # Generate realistic price data
    price = 116000 + np.cumsum(np.random.randn(n_rows) * 50)
    
    data = {
        'timestamp': dates,
        'open': price + np.random.randn(n_rows) * 20,
        'high': price + np.abs(np.random.randn(n_rows) * 30),
        'low': price - np.abs(np.random.randn(n_rows) * 30),
        'close': price,
        'volume': np.random.rand(n_rows) * 10,
        'swing_type': [None] * n_rows,
        'raw_label': np.random.randint(0, 2, n_rows, p=[0.9, 0.1]),
        'confirmed_label': np.random.randint(0, 2, n_rows, p=[0.9, 0.1]),
        'future_move_pct': np.random.rand(n_rows) * 2,
        'is_confirmed_reversal': np.random.rand(n_rows) < 0.1
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    features_df = main()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
