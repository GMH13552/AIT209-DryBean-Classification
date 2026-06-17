"""
Dry Bean Classification - Main Entry Point
Orchestrates: data loading -> preprocessing -> training -> evaluation
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import preprocess_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


def measure_speed(model, X, n_runs=100):
    """Measure inference speed in milliseconds."""
    _ = model.predict(X[:1])  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X)
    return (time.perf_counter() - t0) / n_runs * 1000


def build_models():
    """Return list of (name, model) tuples."""
    return [
        ("KNN (k=5)", 
         KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
         "distance-based"),
        ("SVM (RBF, C=10)", 
         SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
         "kernel-method"),
        ("Random Forest (100 trees)", 
         RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
         "bagging"),
        ("AdaBoost (50 rounds)", 
         AdaBoostClassifier(
             estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
             n_estimators=50, random_state=42
         ),
         "boosting-extra"),  # 课外自学
    ]


def main():
    print("=" * 72)
    print("  Dry Bean Classification - Full Pipeline")
    print("  AIT209 机器学习与项目实践 · 期末作业")
    print("=" * 72)
    
    # ── 1. Load & Preprocess ──
    print("\n[1/3] Loading & preprocessing data ...")
    t0 = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = \
        preprocess_pipeline()
    prep_time = time.perf_counter() - t0
    print(f"  Done in {prep_time:.1f}s")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {le.classes_.tolist()}")
    
    # ── 2. Train ──
    print("\n[2/3] Training models ...")
    models = build_models()
    results = []
    
    for name, model, category in models:
        print(f"\n  -- {name} [{category}] --")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)
        y_test_pred  = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc   = accuracy_score(y_val, y_val_pred)
        test_acc  = accuracy_score(y_test, y_test_pred)
        test_f1   = f1_score(y_test, y_test_pred, average='weighted')
        test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_rec  = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        overfit   = train_acc - val_acc
        
        speed = measure_speed(model, X_test)
        
        print(f"    Train Acc : {train_acc:.4f}")
        print(f"    Val Acc   : {val_acc:.4f}")
        print(f"    Test Acc  : {test_acc:.4f}")
        print(f"    F1        : {test_f1:.4f}")
        print(f"    Overfit   : {overfit:+.4f}")
        print(f"    Speed     : {speed:.1f} ms")
        print(f"    Time      : {train_time:.1f}s")
        
        results.append({
            'name': name, 'category': category,
            'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc,
            'f1': test_f1, 'precision': test_prec, 'recall': test_rec,
            'overfit_gap': overfit, 'speed_ms': speed, 'train_time_s': train_time
        })
    
    # ── 3. Report ──
    print("\n" + "=" * 72)
    print("[3/3] Final Report")
    print("=" * 72)
    
    df = pd.DataFrame(results)
    df['rank'] = df['test_acc'].rank(ascending=False).astype(int)
    df = df.sort_values('test_acc', ascending=False)
    
    RESULTS_DIR = ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    
    # Pretty print
    header = f"{'Rank':<5} {'Algorithm':<28} {'Test Acc':<10} {'F1':<10} {'Overfit':<10} {'Speed':<12} {'Time':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for _, row in df.iterrows():
        print(f"{row['rank']:<5} {row['name']:<28} {row['test_acc']:.4f}     {row['f1']:.4f}     {row['overfit_gap']:+.4f}     {row['speed_ms']:.1f} ms     {row['train_time_s']:.1f}s")
    
    best = df.iloc[0]
    print(f"\n  Best Model: {best['name']} → Test Accuracy = {best['test_acc']:.4f}")
    
    print(f"\n  Results saved to: {RESULTS_DIR / 'model_comparison.csv'}")
    print("=" * 72)
    
    return df


if __name__ == "__main__":
    main()
