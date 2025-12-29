"""
Phase 3: 特徵探索 & 公式發現
===========================================
目標：單獨的下探索不同争幢上的特徵組合
使用：
1. 特徵重要性批斷 (Permutation Importance)
2. 決策樹見親性 (Decision Tree Rules)
3. 符號回歸 (SymPy / PySR) - 需要時間
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier, export_text
import warnings

warnings.filterwarnings('ignore')


class FeatureExplorer:
    """
    特徵探索 & 公式發現類
    """
    
    def __init__(self, samples_df: pd.DataFrame):
        """
        參數：
        - samples_df: Phase 2.5 輸出的 samples DataFrame
        """
        self.df = samples_df.copy()
        
        # 樣本列 (X) 並標籤 (y)
        # 排除 sample_id, timestamp, swing_type, reversal_strength, OHLCV
        exclude_cols = ['sample_id', 'timestamp', 'swing_type', 'candle_index',
                       'open', 'high', 'low', 'close', 'volume',
                       'reversal_strength', 'is_reversal']
        
        self.feature_cols = [col for col in self.df.columns 
                            if col not in exclude_cols]
        
        self.X = self.df[self.feature_cols].fillna(0)  # NaN 填 0
        self.y = self.df['is_reversal'].astype(int)
        
        # 一次性
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 模型
        self.model = None
        self.decision_tree = None
    
    def train_random_forest(self, test_size=0.2, random_state=42):
        """
        訓練 Random Forest 並計算特徵重要性
        """
        print(f"\n訓練 Random Forest...")
        
        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        # 訓練
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # 計算性能
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
        }
        
        print(f"\n模型性能:")
        print(f"  準確率: {results['accuracy']:.4f}")
        print(f"  精步: {results['precision']:.4f}")
        print(f"  召回: {results['recall']:.4f}")
        print(f"  F1 估數: {results['f1']:.4f}")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}" if not np.isnan(results['roc_auc']) else "  ROC-AUC: 沒有訓練")
        
        return results
    
    def get_feature_importance(self, top_n=15) -> pd.DataFrame:
        """
        取得特徵重要性排序
        """
        if self.model is None:
            raise ValueError("轉先需要訓練模型！")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n最重要的 {top_n} 個特徵:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)
    
    def train_decision_tree(self, max_depth=5, random_state=42):
        """
        訓練決策樹並提取見親的規則
        """
        print(f"\n訓練決策樹 (max_depth={max_depth})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y,
            test_size=0.2,
            random_state=random_state,
            stratify=self.y
        )
        
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state
        )
        self.decision_tree.fit(X_train, y_train)
        
        # 計算性能
        y_pred = self.decision_tree.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  準確率: {acc:.4f}")
        
        # 提取見親的规則
        tree_rules = export_text(
            self.decision_tree,
            feature_names=self.feature_cols,
            max_depth=max_depth
        )
        
        print(f"\n見親的决策规则:")
        print(tree_rules)
        
        return tree_rules
    
    def get_feature_interactions(self, top_features: list, top_n=10) -> pd.DataFrame:
        """
        探索特徵之間的相關性
        """
        print(f"\n特徵之間的相關性 (反轉 vs 非反轉):")
        
        interactions = []
        
        for feat in top_features:
            if feat in self.df.columns:
                # 計算有效的那佋
                reversal_mean = self.df[self.df['is_reversal'] == 1][feat].mean()
                non_reversal_mean = self.df[self.df['is_reversal'] == 0][feat].mean()
                
                # 面積並計算䗢每每比樣
                if non_reversal_mean != 0:
                    ratio = reversal_mean / non_reversal_mean
                else:
                    ratio = np.nan
                
                interactions.append({
                    'feature': feat,
                    'reversal_mean': reversal_mean,
                    'non_reversal_mean': non_reversal_mean,
                    'mean_ratio': ratio,
                    'difference': reversal_mean - non_reversal_mean,
                })
        
        interactions_df = pd.DataFrame(interactions).sort_values(
            'mean_ratio', ascending=False, key=abs
        )
        
        print(interactions_df.head(top_n).to_string(index=False))
        
        return interactions_df
    
    def export_summary(self, output_file: str = None):
        """
        導出整简接
        """
        if output_file is None:
            output_file = 'phase3_exploration_summary.txt'
        
        with open(output_file, 'w') as f:
            f.write("""
========================================
Phase 3: 特徵探索結果粗殶
========================================

1. 特徵重要性（Random Forest）
2. 決策规則（Decision Tree）
3. 特徵相關性（反轉 vs 非反轉）

這些特徵的組合可以策創優斯回歸公式！
""")
        
        print(f"\n✓ 整简已導出: {output_file}")


def main():
    """
    主函數：執行 Phase 3 特徵探索
    """
    import sys
    
    pair = "BTCUSDT"
    timeframe = "15m"
    samples_file = f"{pair}_{timeframe}_samples.csv"
    
    print(f"Phase 3: 特徵探索")
    print("=" * 70)
    print(f"配對: {pair}")
    print(f"時間框架: {timeframe}")
    print("=" * 70)
    
    # 加載樣本
    print(f"\n加載: {samples_file}")
    try:
        samples_df = pd.read_csv(samples_file)
        print(f"  樣本數: {len(samples_df)}")
    except FileNotFoundError:
        print(f"錯誤：找不到 {samples_file}")
        sys.exit(1)
    
    # 初始化探索器
    explorer = FeatureExplorer(samples_df)
    
    # 1. Random Forest 訓練
    print("\n" + "="*70)
    print("步驟 1: Random Forest 訓練")
    print("="*70)
    results = explorer.train_random_forest()
    
    # 2. 特徵重要性
    print("\n" + "="*70)
    print("步驟 2: 特徵重要性分析")
    print("="*70)
    importance_df = explorer.get_feature_importance(top_n=15)
    importance_df.to_csv(f'{pair}_{timeframe}_feature_importance.csv', index=False)
    print(f"  ✓ 保存: {pair}_{timeframe}_feature_importance.csv")
    
    # 3. 決策樹見親规則
    print("\n" + "="*70)
    print("步驟 3: 決策樹見親规則")
    print("="*70)
    tree_rules = explorer.train_decision_tree(max_depth=5)
    
    # 保存规則
    with open(f'{pair}_{timeframe}_tree_rules.txt', 'w') as f:
        f.write(tree_rules)
    print(f"  ✓ 保存: {pair}_{timeframe}_tree_rules.txt")
    
    # 4. 特徵相關性
    print("\n" + "="*70)
    print("步驟 4: 特徵相關性分析")
    print("="*70)
    top_features = importance_df['feature'].head(5).tolist()
    interactions_df = explorer.get_feature_interactions(
        top_features, top_n=10
    )
    interactions_df.to_csv(
        f'{pair}_{timeframe}_feature_interactions.csv', index=False
    )
    print(f"  ✓ 保存: {pair}_{timeframe}_feature_interactions.csv")
    
    # 整简
    explorer.export_summary(f'{pair}_{timeframe}_exploration_summary.txt')
    
    print("\n" + "="*70)
    print("✅ Phase 3 完成！")
    print("="*70)
    print("\
輸出物：
1. {}_feature_importance.csv
2. {}_tree_rules.txt
3. {}_feature_interactions.csv
4. {}_exploration_summary.txt
這些特徵組合會是你的反轉䮤易信號！
".format(pair + '_' + timeframe, pair + '_' + timeframe,
            pair + '_' + timeframe, pair + '_' + timeframe))


if __name__ == "__main__":
    main()
