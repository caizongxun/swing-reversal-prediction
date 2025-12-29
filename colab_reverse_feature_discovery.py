"""
COMPLETE COLAB WORKFLOW: Reverse Feature Discovery for Swing Reversal Prediction

Phase 1: Load 10,000 K棒 data and identify confirmed reversals
Phase 2: Extract mathematical features around those reversals
Phase 3: Identify key patterns differentiating true vs false signals
Phase 4: Generate formula candidates for AI to learn
Phase 5: Export results for training

Usage in Colab:
!git clone https://github.com/caizongxun/swing-reversal-prediction.git
%cd swing-reversal-prediction
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
!python colab_reverse_feature_discovery.py --data_path {filename} --max_rows 10000
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from phase1_analysis_10k import load_data, print_summary as print_phase1_summary
    from swing_reversal_detector import SwingReversalDetector
    from reverse_feature_extraction import ReverseFeatureExtractor, TechnicalIndicators
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the same directory")
    sys.exit(1)


class ReverseFeatureDiscoveryPipeline:
    """
    Complete pipeline for discovering mathematical formulas from reversals.
    """
    
    def __init__(self, max_rows: int = 10000, window: int = 5, future_candles: int = 12, threshold: float = 1.0):
        self.max_rows = max_rows
        self.window = window
        self.future_candles = future_candles
        self.threshold = threshold
        self.detector = SwingReversalDetector(window, future_candles, threshold)
        self.extractor = ReverseFeatureExtractor(lookback=20)
    
    def phase1_detect_reversals(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Phase 1: Load data and detect swing reversals
        """
        print("\n" + "="*80)
        print("PHASE 1: SWING REVERSAL DETECTION (10,000+ Candles)")
        print("="*80)
        
        df = load_data(csv_path, max_rows=self.max_rows)
        print(f"Loaded {len(df)} candles")
        
        # Detect reversals
        df, stats = self.detector.process(df)
        
        print(f"\nDetection Results:")
        print(f"  Raw Swing Points: {stats['total_raw_swings']:,}")
        print(f"  Confirmed Reversals: {stats['total_confirmed_reversals']:,}")
        print(f"  True Reversal Rate: {stats['filtering_ratio']:.2%}")
        
        return df, stats
    
    def phase2_extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: Extract mathematical features from reversals
        """
        print("\n" + "="*80)
        print("PHASE 2: MATHEMATICAL FEATURE EXTRACTION")
        print("="*80)
        
        confirmed = df[df['confirmed_label'] == 1]
        print(f"Extracting features from {len(confirmed)} confirmed reversals...")
        
        features_df = self.extractor.extract_all_reversals(df)
        print(f"Extracted {len(features_df)} feature sets")
        
        # Display sample features
        print(f"\nFeature Columns Extracted:")
        feature_cols = [col for col in features_df.columns if col not in ['reversal_type', 'future_move', 'confirmed']]
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        return features_df
    
    def phase3_identify_patterns(self, features_df: pd.DataFrame) -> Dict:
        """
        Phase 3: Identify key patterns
        """
        print("\n" + "="*80)
        print("PHASE 3: PATTERN IDENTIFICATION")
        print("="*80)
        
        patterns = self.extractor.identify_key_patterns(features_df)
        
        print(f"\nTop 10 Key Patterns (sorted by difference ratio):")
        print(f"{'Rank':<5} {'Feature':<25} {'Confirmed':<12} {'False':<12} {'Diff Ratio':<12}")
        print(f"{'-'*66}")
        
        for i, (feature, info) in enumerate(list(patterns.items())[:10], 1):
            confirmed_mean = info['confirmed_mean']
            false_mean = info['false_mean']
            diff_ratio = info['difference_ratio']
            print(f"{i:<5} {feature:<25} {confirmed_mean:<12.4f} {false_mean:<12.4f} {diff_ratio:<12.4f}")
        
        return patterns
    
    def phase4_generate_formulas(self, patterns: Dict) -> list:
        """
        Phase 4: Generate formula candidates
        """
        print("\n" + "="*80)
        print("PHASE 4: FORMULA GENERATION")
        print("="*80)
        
        formulas = self.extractor.generate_formula_candidates(patterns, top_n=8)
        
        print(f"\nGenerated {len(formulas)} formula candidates:")
        print(f"\nNumerical Formulas:")
        numerical = [f for f in formulas if '>' not in f and '<' not in f and 'AND' not in f]
        for i, formula in enumerate(numerical[:10], 1):
            print(f"  {i:2d}. {formula}")
        
        print(f"\nLogical Conditions:")
        logical = [f for f in formulas if '>' in f or '<' in f or 'AND' in f]
        for i, formula in enumerate(logical, 1):
            print(f"  {i:2d}. {formula}")
        
        return formulas
    
    def phase5_export_results(self, df: pd.DataFrame, features_df: pd.DataFrame, 
                             patterns: Dict, formulas: list, output_prefix: str = "reversal") -> None:
        """
        Phase 5: Export all results
        """
        print("\n" + "="*80)
        print("PHASE 5: EXPORT RESULTS")
        print("="*80)
        
        # Export labeled data
        labeled_file = f"{output_prefix}_labeled_data_10k.csv"
        export_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'swing_type', 'raw_label', 'confirmed_label', 
                      'future_move_pct', 'is_confirmed_reversal']
        export_cols = [col for col in export_cols if col in df.columns]
        df[export_cols].to_csv(labeled_file, index=False)
        print(f"Exported labeled data to {labeled_file}")
        
        # Export features
        features_file = f"{output_prefix}_extracted_features_10k.csv"
        features_df.to_csv(features_file, index=False)
        print(f"Exported features to {features_file}")
        
        # Export patterns report
        patterns_file = f"{output_prefix}_pattern_analysis_10k.txt"
        with open(patterns_file, 'w') as f:
            f.write("IDENTIFIED KEY PATTERNS FOR REVERSAL PREDICTION\n")
            f.write("="*70 + "\n\n")
            
            for feature, info in list(patterns.items())[:20]:
                f.write(f"{feature}:\n")
                f.write(f"  Confirmed Mean: {info['confirmed_mean']:.6f}\n")
                f.write(f"  False Mean: {info['false_mean']:.6f}\n")
                f.write(f"  Difference Ratio: {info['difference_ratio']:.6f}\n")
                f.write(f"  Better for Confirmation: {info['better_for_confirmation']}\n\n")
        
        print(f"Exported patterns to {patterns_file}")
        
        # Export formulas
        formulas_file = f"{output_prefix}_formula_candidates_10k.txt"
        with open(formulas_file, 'w') as f:
            f.write("AI-GENERATED FORMULA CANDIDATES FOR REVERSAL PREDICTION\n")
            f.write("="*70 + "\n\n")
            f.write("These formulas are candidates that the AI model will learn to validate.\n\n")
            
            f.write("NUMERICAL FORMULAS (for regression/scoring):\n")
            f.write("-"*70 + "\n")
            numerical = [f for f in formulas if '>' not in f and '<' not in f and 'AND' not in f]
            for i, formula in enumerate(numerical, 1):
                f.write(f"{i:2d}. {formula}\n")
            
            f.write("\nLOGICAL CONDITIONS (for classification):\n")
            f.write("-"*70 + "\n")
            logical = [f for f in formulas if '>' in f or '<' in f or 'AND' in f]
            for i, formula in enumerate(logical, 1):
                f.write(f"{i:2d}. {formula}\n")
        
        print(f"Exported formulas to {formulas_file}")
        
        # Summary statistics
        summary_file = f"{output_prefix}_discovery_summary_10k.txt"
        with open(summary_file, 'w') as f:
            f.write("REVERSE FEATURE DISCOVERY - EXECUTION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {len(df)} candles analyzed\n")
            f.write(f"Raw Swing Points Detected: {len(df[df['raw_label']==1])}\n")
            f.write(f"Confirmed Reversals: {len(df[df['confirmed_label']==1])}\n")
            f.write(f"True Reversal Rate: {len(df[df['confirmed_label']==1]) / len(df[df['raw_label']==1]) * 100:.2f}%\n")
            f.write(f"\nFeatures Extracted: {len(features_df.columns) - 3}\n")
            f.write(f"Key Patterns Identified: {len(patterns)}\n")
            f.write(f"Formula Candidates Generated: {len(formulas)}\n")
            f.write(f"\nNext Step: Use these formulas for AI model training\n")
        
        print(f"Exported summary to {summary_file}")
        print(f"\nAll files ready for download from Colab!")
    
    def run_complete_pipeline(self, csv_path: str, output_prefix: str = "reversal") -> None:
        """
        Run the complete discovery pipeline
        """
        print("\n" + "#"*80)
        print("# REVERSE FEATURE DISCOVERY PIPELINE FOR SWING REVERSAL PREDICTION")
        print("#"*80)
        
        # Phase 1
        df, stats = self.phase1_detect_reversals(csv_path)
        
        # Phase 2
        features_df = self.phase2_extract_features(df)
        
        # Phase 3
        patterns = self.phase3_identify_patterns(features_df)
        
        # Phase 4
        formulas = self.phase4_generate_formulas(patterns)
        
        # Phase 5
        self.phase5_export_results(df, features_df, patterns, formulas, output_prefix)
        
        print("\n" + "#"*80)
        print("# PIPELINE COMPLETE!")
        print("#"*80)
        print(f"\nReady to download:")
        print(f"  - {output_prefix}_labeled_data_10k.csv")
        print(f"  - {output_prefix}_extracted_features_10k.csv")
        print(f"  - {output_prefix}_pattern_analysis_10k.txt")
        print(f"  - {output_prefix}_formula_candidates_10k.txt")
        print(f"  - {output_prefix}_discovery_summary_10k.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Reverse Feature Discovery Pipeline for Colab'
    )
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to K-line data CSV')
    parser.add_argument('--max_rows', type=int, default=10000,
                       help='Maximum rows to analyze (default: 10000)')
    parser.add_argument('--window', type=int, default=5,
                       help='Swing detection window (default: 5)')
    parser.add_argument('--future_candles', type=int, default=12,
                       help='Future candles to check (default: 12)')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='Reversal confirmation threshold % (default: 1.0)')
    parser.add_argument('--output_prefix', type=str, default='reversal',
                       help='Output file prefix (default: reversal)')
    
    args = parser.parse_args()
    
    pipeline = ReverseFeatureDiscoveryPipeline(
        max_rows=args.max_rows,
        window=args.window,
        future_candles=args.future_candles,
        threshold=args.threshold
    )
    
    pipeline.run_complete_pipeline(args.data_path, args.output_prefix)


if __name__ == '__main__':
    main()
