"""
Phase 1 Example Usage Patterns
Demonstrates various ways to use the swing reversal detector.

Run individual examples to understand the library better.
"""

import pandas as pd
from swing_reversal_detector import SwingReversalDetector
import matplotlib.pyplot as plt


def load_sample_data(filepath):
    """Load CSV data and prepare for analysis."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    if 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df.reset_index(drop=True)


def example_1_basic_detection():
    """
    Example 1: Basic Swing Detection
    Using default parameters (window=5, future_candles=12, threshold=1.0)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Swing Detection")
    print("="*70)
    
    # Load data
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    print(f"Loaded {len(df)} candles")
    
    # Initialize detector with defaults
    detector = SwingReversalDetector()
    
    # Run pipeline
    results_df, stats = detector.process(df)
    
    # Print summary
    print(f"\nTotal Raw Swings: {stats['total_raw_swings']}")
    print(f"Confirmed Reversals: {stats['total_confirmed_reversals']}")
    print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
    print(f"\nSwing Highs: {stats['swing_highs_detected']}")
    print(f"  \u2514 Confirmed: {stats['confirmed_highs']} ({stats['high_confirmation_rate']:.1%})")
    print(f"\nSwing Lows: {stats['swing_lows_detected']}")
    print(f"  \u2514 Confirmed: {stats['confirmed_lows']} ({stats['low_confirmation_rate']:.1%})")
    
    return results_df, stats


def example_2_strict_filtering():
    """
    Example 2: Strict Filtering (Reduce False Signals)
    Higher threshold = fewer but higher-quality signals
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Strict Filtering (Threshold=1.5%, Window=7)")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    # More strict parameters
    detector = SwingReversalDetector(
        window=7,
        future_candles=12,
        move_threshold=1.5
    )
    
    results_df, stats = detector.process(df)
    
    print(f"\nRaw Swings: {stats['total_raw_swings']}")
    print(f"Confirmed Reversals: {stats['total_confirmed_reversals']}")
    print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
    print(f"\nNote: Fewer confirmations but higher-quality signals")
    
    # Show avg movements
    confirmed = results_df[results_df['confirmed_label'] == 1]
    false_sigs = results_df[(results_df['raw_label'] == 1) & (results_df['confirmed_label'] == 0)]
    
    print(f"\nConfirmed Avg Move: {confirmed['future_move_pct'].mean():.3f}%")
    print(f"False Signal Avg Move: {false_sigs['future_move_pct'].mean():.3f}%")
    
    return results_df, stats


def example_3_sensitive_detection():
    """
    Example 3: Sensitive Detection (Capture More Opportunities)
    Lower threshold and window = more signals, potentially more false positives
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Sensitive Detection (Threshold=0.8%, Window=3)")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    # More sensitive parameters
    detector = SwingReversalDetector(
        window=3,
        future_candles=12,
        move_threshold=0.8
    )
    
    results_df, stats = detector.process(df)
    
    print(f"\nRaw Swings: {stats['total_raw_swings']}")
    print(f"Confirmed Reversals: {stats['total_confirmed_reversals']}")
    print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
    print(f"\nNote: More confirmations but lower average quality")
    
    confirmed = results_df[results_df['confirmed_label'] == 1]
    print(f"\nConfirmed Avg Move: {confirmed['future_move_pct'].mean():.3f}%")
    
    return results_df, stats


def example_4_scalping_setup():
    """
    Example 4: Scalping Setup
    Shorter confirmation window (6 candles = 1.5 hours on 15m)
    Lower threshold for faster entries
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Scalping Setup (6 candles, 0.8% threshold)")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    detector = SwingReversalDetector(
        window=5,
        future_candles=6,      # 1.5 hours
        move_threshold=0.8     # Lower bar
    )
    
    results_df, stats = detector.process(df)
    
    print(f"\nRaw Swings: {stats['total_raw_swings']}")
    print(f"Confirmed Reversals: {stats['total_confirmed_reversals']}")
    print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
    print(f"\nIdeal for: Fast-paced scalping, frequent entries")
    
    return results_df, stats


def example_5_swing_trading_setup():
    """
    Example 5: Swing Trading Setup
    Longer confirmation window (20 candles = 5 hours on 15m)
    Higher threshold for fewer, better-quality signals
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Swing Trading Setup (20 candles, 1.2% threshold)")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    detector = SwingReversalDetector(
        window=5,
        future_candles=20,     # 5 hours
        move_threshold=1.2     # Higher bar
    )
    
    results_df, stats = detector.process(df)
    
    print(f"\nRaw Swings: {stats['total_raw_swings']}")
    print(f"Confirmed Reversals: {stats['total_confirmed_reversals']}")
    print(f"Filtering Ratio: {stats['filtering_ratio']:.2%}")
    print(f"\nIdeal for: Swing trading, fewer entries but better quality")
    
    return results_df, stats


def example_6_export_and_analyze():
    """
    Example 6: Export Data for Phase 2
    Save labeled data to CSV for feature engineering and model training
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Data for Phase 2")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    detector = SwingReversalDetector()
    results_df, stats = detector.process(df)
    
    # Export
    export_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                  'swing_type', 'raw_label', 'confirmed_label', 
                  'future_move_pct', 'is_confirmed_reversal']
    export_cols = [col for col in export_cols if col in results_df.columns]
    results_df[export_cols].to_csv('phase1_labeled_data.csv', index=False)
    
    print(f"\nExported {len(results_df)} rows to 'phase1_labeled_data.csv'")
    
    # Show sample
    confirmed = results_df[results_df['confirmed_label'] == 1]
    print(f"\nSample of confirmed reversals:")
    print(confirmed[['timestamp', 'close', 'swing_type', 'future_move_pct']].head(10))
    
    # Statistics for Phase 2
    print(f"\nFor Phase 2 (Feature Engineering):")
    print(f"  - Total samples: {len(results_df)}")
    print(f"  - Positive class (True Reversals): {confirmed['confirmed_label'].sum()}")
    print(f"  - Negative class (False Signals): {(results_df['raw_label'] == 1).sum() - confirmed['confirmed_label'].sum()}")
    print(f"  - Class balance: {confirmed['confirmed_label'].sum() / (results_df['raw_label'] == 1).sum():.1%} positive")
    
    return results_df


def example_7_compare_parameters():
    """
    Example 7: Parameter Comparison
    Shows how different parameters affect results
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Parameter Sensitivity Analysis")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    # Test different thresholds
    thresholds = [0.5, 0.8, 1.0, 1.5, 2.0]
    print("\nThreshold Sensitivity (fixed window=5, future_candles=12):")
    print("Threshold | Raw Swings | Confirmed | Ratio | Avg Move")
    print("-" * 60)
    
    for thresh in thresholds:
        detector = SwingReversalDetector(
            window=5,
            future_candles=12,
            move_threshold=thresh
        )
        _, stats = detector.process(df)
        print(f"  {thresh:0.1f}%    | {stats['total_raw_swings']:3d}        | "
              f"{stats['total_confirmed_reversals']:3d}        | "
              f"{stats['filtering_ratio']:0.1%}   | "
              f"{stats['total_confirmed_reversals'] / max(stats['total_raw_swings'], 1) * 1.45:.2f}%")
    
    # Test different windows
    windows = [3, 5, 7, 10]
    print("\nWindow Size Sensitivity (fixed threshold=1.0%, future_candles=12):")
    print("Window | Raw Swings | Confirmed | Ratio")
    print("-" * 45)
    
    for w in windows:
        detector = SwingReversalDetector(
            window=w,
            future_candles=12,
            move_threshold=1.0
        )
        _, stats = detector.process(df)
        print(f"  {w:2d}   | {stats['total_raw_swings']:3d}        | "
              f"{stats['total_confirmed_reversals']:3d}        | "
              f"{stats['filtering_ratio']:0.1%}")


def example_8_signal_quality_analysis():
    """
    Example 8: Signal Quality Analysis
    Compare confirmed reversals vs. false signals in detail
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Signal Quality Analysis")
    print("="*70)
    
    df = load_sample_data('BTCUSDT_15m_binance_us.csv')
    
    detector = SwingReversalDetector()
    results_df, stats = detector.process(df)
    
    confirmed = results_df[results_df['confirmed_label'] == 1]
    false_sigs = results_df[(results_df['raw_label'] == 1) & (results_df['confirmed_label'] == 0)]
    
    print(f"\nConfirmed Reversals (TRUE signals):")
    print(f"  Count: {len(confirmed)}")
    print(f"  Avg Movement: {confirmed['future_move_pct'].mean():.3f}%")
    print(f"  Std Deviation: {confirmed['future_move_pct'].std():.3f}%")
    print(f"  Min: {confirmed['future_move_pct'].min():.3f}%")
    print(f"  Max: {confirmed['future_move_pct'].max():.3f}%")
    print(f"  Median: {confirmed['future_move_pct'].median():.3f}%")
    
    print(f"\nFalse Signals (Filtered OUT):")
    print(f"  Count: {len(false_sigs)}")
    print(f"  Avg Movement: {false_sigs['future_move_pct'].mean():.3f}%")
    print(f"  Std Deviation: {false_sigs['future_move_pct'].std():.3f}%")
    print(f"  Min: {false_sigs['future_move_pct'].min():.3f}%")
    print(f"  Max: {false_sigs['future_move_pct'].max():.3f}%")
    print(f"  Median: {false_sigs['future_move_pct'].median():.3f}%")
    
    print(f"\nSignal Separation Quality:")
    avg_gap = confirmed['future_move_pct'].mean() - false_sigs['future_move_pct'].mean()
    print(f"  Gap between averages: {avg_gap:.3f}%")
    print(f"  Separation ratio: {confirmed['future_move_pct'].mean() / false_sigs['future_move_pct'].mean():.1f}x")
    
    # Plot histogram
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(confirmed['future_move_pct'], bins=20, alpha=0.7, label='Confirmed', color='green')
        plt.hist(false_sigs['future_move_pct'], bins=20, alpha=0.7, label='False Signals', color='red')
        plt.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel('Future Movement %')
        plt.ylabel('Frequency')
        plt.title('Signal Quality Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('signal_quality_distribution.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to 'signal_quality_distribution.png'")
    except ImportError:
        print("\n(Matplotlib not available, skipping visualization)")


if __name__ == '__main__':
    """
    Run examples
    Uncomment the examples you want to run
    """
    
    print("\n" + "#" * 70)
    print("# Swing Reversal Detector - Phase 1 Examples")
    print("#" * 70)
    
    # Example 1: Basic detection
    example_1_basic_detection()
    
    # Example 2: Strict filtering
    # example_2_strict_filtering()
    
    # Example 3: Sensitive detection  
    # example_3_sensitive_detection()
    
    # Example 4: Scalping setup
    # example_4_scalping_setup()
    
    # Example 5: Swing trading setup
    # example_5_swing_trading_setup()
    
    # Example 6: Export for Phase 2
    # example_6_export_and_analyze()
    
    # Example 7: Parameter sensitivity
    # example_7_compare_parameters()
    
    # Example 8: Signal quality analysis
    # example_8_signal_quality_analysis()
    
    print("\n" + "#" * 70)
    print("# Examples completed")
    print("#" * 70 + "\n")
