"""
Phase 1 Analysis Pipeline - Extended for 10,000+ Candles
Optimized for large-scale swing reversal detection with intelligent filtering.

Usage:
    python phase1_analysis_10k.py --data_path BTCUSDT_15m_binance_us.csv --window 5 --future_candles 12 --threshold 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import argparse
from pathlib import Path
from swing_reversal_detector import SwingReversalDetector
import warnings
warnings.filterwarnings('ignore')


def load_data(csv_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load OHLCV data from Binance US CSV.
    Optimized for large datasets (10000+ rows).
    
    Parameters:
    - csv_path: Path to CSV file
    - max_rows: Maximum rows to load (None = all rows)
    """
    print(f"Loading data from {csv_path}...")
    
    if max_rows:
        df = pd.read_csv(csv_path, nrows=max_rows)
    else:
        df = pd.read_csv(csv_path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    column_mapping = {
        'open_time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
    
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} candles")
    return df


def plot_results_10k(df: pd.DataFrame, output_path: str = 'swing_reversal_analysis_10k.png'):
    """
    Optimized visualization for large datasets (10k+ candles).
    Uses sampling to avoid overplotting.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Use sampling for plotting efficiency
    sample_size = min(2000, len(df))
    plot_df = df.iloc[::max(1, len(df)//sample_size)].reset_index(drop=True)
    x_axis = np.arange(len(plot_df))
    
    # Plot 1: Price and Swing Points
    ax1.plot(x_axis, plot_df['close'], linewidth=1.5, label='Close Price', 
            color='black', zorder=1, alpha=0.8)
    
    # Plot swing points (downsampled)
    raw_highs = plot_df[(plot_df['swing_type'] == 'high') & (plot_df['raw_label'] == 1)]
    if len(raw_highs) > 0:
        ax1.scatter(raw_highs.index, raw_highs['close'], 
                   marker='o', s=50, color='blue', alpha=0.5, 
                   label=f'Raw Swing Highs ({len(raw_highs)})', zorder=3)
    
    raw_lows = plot_df[(plot_df['swing_type'] == 'low') & (plot_df['raw_label'] == 1)]
    if len(raw_lows) > 0:
        ax1.scatter(raw_lows.index, raw_lows['close'], 
                   marker='o', s=50, color='lightblue', alpha=0.5, 
                   label=f'Raw Swing Lows ({len(raw_lows)})', zorder=3)
    
    # Plot confirmed reversals
    confirmed_highs = plot_df[(plot_df['swing_type'] == 'high') & 
                              (plot_df['confirmed_label'] == 1)]
    if len(confirmed_highs) > 0:
        ax1.scatter(confirmed_highs.index, confirmed_highs['close'], 
                   marker='v', s=100, color='green', alpha=0.8, 
                   label=f'Confirmed High ({len(confirmed_highs)})', zorder=5, 
                   edgecolors='darkgreen', linewidths=1.5)
    
    confirmed_lows = plot_df[(plot_df['swing_type'] == 'low') & 
                             (plot_df['confirmed_label'] == 1)]
    if len(confirmed_lows) > 0:
        ax1.scatter(confirmed_lows.index, confirmed_lows['close'], 
                   marker='^', s=100, color='lime', alpha=0.8, 
                   label=f'Confirmed Low ({len(confirmed_lows)})', zorder=5, 
                   edgecolors='darkgreen', linewidths=1.5)
    
    ax1.set_title('Swing Reversal Detection - 10K+ Candles Analysis', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Future Price Movement Distribution
    swing_mask = plot_df['raw_label'] == 1
    if swing_mask.sum() > 0:
        future_moves = plot_df[swing_mask]['future_move_pct']
        colors = ['green' if x >= 1.0 else 'red' for x in future_moves]
        ax2.bar(plot_df[swing_mask].index, future_moves, 
               color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                   label='Confirmation Threshold (1.0%)')
        ax2.set_ylabel('Future Move % (12 candles)', fontsize=11)
        ax2.set_xlabel('Candle Index (downsampled)', fontsize=11)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def print_summary(df: pd.DataFrame, stats: dict, detector_params: dict):
    """
    Print comprehensive summary for 10k+ analysis.
    """
    print("\n" + "="*80)
    print("SWING REVERSAL DETECTION - PHASE 1 (10K+ CANDLES) ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nDETECTOR PARAMETERS:")
    print(f"  Window Size: {detector_params['window']}")
    print(f"  Future Candles: {detector_params['future_candles']}")
    print(f"  Move Threshold: {detector_params['move_threshold']}%")
    
    print(f"\nDATASET STATISTICS:")
    print(f"  Total Candles: {len(df):,}")
    if 'timestamp' in df.columns:
        print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  Price Volatility: ${df['high'].max() - df['low'].min():.2f}")
    
    print(f"\nSWING DETECTION RESULTS:")
    print(f"  Total Raw Swing Points: {stats['total_raw_swings']:,}")
    print(f"  ├─ Swing Highs: {stats['swing_highs_detected']:,}")
    print(f"  └─ Swing Lows: {stats['swing_lows_detected']:,}")
    print(f"  Raw Swing Density: {stats['total_raw_swings']/len(df)*100:.2f}% of candles")
    
    print(f"\nINTELLIGENT FILTERING RESULTS:")
    print(f"  Confirmed True Reversals: {stats['total_confirmed_reversals']:,}")
    print(f"  ├─ Confirmed Highs: {stats['confirmed_highs']:,}")
    print(f"  └─ Confirmed Lows: {stats['confirmed_lows']:,}")
    print(f"  False Signals Filtered: {stats['total_raw_swings'] - stats['total_confirmed_reversals']:,}")
    print(f"  True Reversal Rate: {stats['filtering_ratio']:.2%}")
    print(f"  Signal Quality: {stats['filtering_ratio']:.3f}")
    
    print(f"\nCONFIRMATION RATES:")
    print(f"  Swing High Confirmation: {stats['high_confirmation_rate']:.2%}")
    print(f"  Swing Low Confirmation: {stats['low_confirmation_rate']:.2%}")
    
    confirmed = df[df['confirmed_label'] == 1]
    false = df[(df['raw_label'] == 1) & (df['confirmed_label'] == 0)]
    
    if len(confirmed) > 0:
        print(f"\nCONFIRMED REVERSALS MOVEMENT:")
        print(f"  Average Future Move: {confirmed['future_move_pct'].mean():.3f}%")
        print(f"  Median Future Move: {confirmed['future_move_pct'].median():.3f}%")
        print(f"  Max Future Move: {confirmed['future_move_pct'].max():.3f}%")
        print(f"  Min Future Move: {confirmed['future_move_pct'].min():.3f}%")
        print(f"  Std Dev: {confirmed['future_move_pct'].std():.3f}%")
    
    if len(false) > 0:
        print(f"\nFALSE SIGNALS MOVEMENT:")
        print(f"  Average Future Move: {false['future_move_pct'].mean():.3f}%")
        print(f"  Median Future Move: {false['future_move_pct'].median():.3f}%")
        print(f"  Max Future Move: {false['future_move_pct'].max():.3f}%")
    
    print(f"\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Extended for 10,000+ Candles Analysis'
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to BTCUSDT CSV file')
    parser.add_argument('--max_rows', type=int, default=10000,
                       help='Maximum rows to load (default: 10000)')
    parser.add_argument('--window', type=int, default=5,
                       help='Window size for swing detection (default: 5)')
    parser.add_argument('--future_candles', type=int, default=12,
                       help='Number of future candles to check (default: 12)')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Price movement threshold % (default: 1.0)')
    parser.add_argument('--output', type=str, default='swing_reversal_analysis_10k.png',
                       help='Output visualization file')
    parser.add_argument('--export_csv', type=str, default=None,
                       help='Export processed data to CSV')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data_path, max_rows=args.max_rows)
    
    # Initialize detector
    detector = SwingReversalDetector(
        window=args.window,
        future_candles=args.future_candles,
        move_threshold=args.threshold
    )
    
    # Run analysis
    print("Detecting swing points...")
    df, stats = detector.process(df)
    
    detector_params = {
        'window': args.window,
        'future_candles': args.future_candles,
        'move_threshold': args.threshold
    }
    print_summary(df, stats, detector_params)
    
    # Generate visualization
    print(f"Generating visualization...")
    plot_results_10k(df, args.output)
    
    # Export CSV
    if args.export_csv:
        print(f"Exporting to {args.export_csv}...")
        export_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'swing_type', 'raw_label', 'confirmed_label', 
                      'future_move_pct', 'is_confirmed_reversal']
        export_cols = [col for col in export_cols if col in df.columns]
        df[export_cols].to_csv(args.export_csv, index=False)
        print(f"Exported {len(df)} rows")
    
    return df, stats


if __name__ == '__main__':
    main()
