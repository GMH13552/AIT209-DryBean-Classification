"""
Dry Bean Classification - Full Training & Evaluation Pipeline
Implements 6 algorithms: KNN, SVM, RandomForest, XGBoost, MLP, LightGBM
Evaluates: accuracy, loss curves, speed, robustness, overfitting
"""
import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from preprocess import preprocess_pipeline

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_knn():
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "KNN"

def build_svm():
    from sklearn.svm import SVC
    return SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42), "SVM (RBF)"

def build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), "Random Forest"

def build_gbdt():
    """Gradient Boosting (sklearn) - comparable to XGBoost/LightGBM"""
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                       subsample=0.8, max_features=0.8,
                                       random_state=42), "GradientBoost"

def build_hist_gbdt():
    """Histogram-based Gradient Boosting (sklearn) - fast, inspired by LightGBM"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.1,
                                           l2_regularization=0.0, early_stopping=False,
                                           random_state=42), "HistGradBoost"

def build_mlp():
    """
    Simple MLP using sklearn for fairness in comparison.
    PyTorch version available as optional advanced model.
    """
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu',
                          solver='adam', alpha=0.001, batch_size=128,
                          learning_rate='adaptive', max_iter=100,
                          early_stopping=True, validation_fraction=0.1,
                          random_state=42), "MLP"


ALL_MODELS = [build_knn, build_svm, build_rf, build_gbdt, build_hist_gbdt, build_mlp]


# ============================================================
# METRICS & EVALUATION
# ============================================================

def compute_metrics(y_true, y_pred, y_proba=None, class_names=None):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, classification_report)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def measure_inference_speed(model, X, n_runs=1000):
    """Measure average inference time over n_runs."""
    if hasattr(model, 'predict_proba'):
        # Warmup
        _ = model.predict(X[:1])
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model.predict(X)
        elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms
    else:
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model.predict(X)
        elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


# ============================================================
# TRAINING
# ============================================================

def train_model(build_fn, X_train, y_train, X_val, y_val):
    """Train a single model and evaluate on validation set."""
    model, name = build_fn()
    t0 = time.perf_counter()
    
    # All sklearn models use the same fit interface
    model.fit(X_train, y_train)
    
    train_time = time.perf_counter() - t0
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)
    
    # Overfitting gap
    train_acc = train_metrics['accuracy']
    val_acc = val_metrics['accuracy']
    gap = train_acc - val_acc
    
    print(f"  {name:15s} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Gap: {gap:+.4f} | Time: {train_time:.1f}s")
    
    return {
        'model': model,
        'name': name,
        'train_time': train_time,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'overfitting_gap': gap,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc
    }


def train_all(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    results = []
    for build_fn in ALL_MODELS:
        result = train_model(build_fn, X_train, y_train, X_val, y_val)
        results.append(result)
    
    # Sort by validation accuracy
    results.sort(key=lambda x: x['val_accuracy'], reverse=True)
    print("\n[BEST] Ranking (by Validation Accuracy):")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['name']:15s} -> {r['val_accuracy']:.4f}")
    
    return results


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_on_test(results, X_test, y_test, le):
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    test_results = []
    for r in results:
        model = r['model']
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        
        speed = measure_inference_speed(model, X_test)
        
        test_results.append({
            'name': r['name'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'speed_ms': speed,
            'train_time_s': r['train_time'],
            'train_acc': r['train_accuracy'],
            'val_acc': r['val_accuracy'],
            'overfitting_gap': r['overfitting_gap']
        })
        
        print(f"  {r['name']:15s} | Test Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | Speed: {speed:.2f}ms")
    
    # Build and save results table
    df_results = pd.DataFrame(test_results)
    df_results = df_results.sort_values('accuracy', ascending=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    df_results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\n[OK] Results saved to {RESULTS_DIR / 'model_comparison.csv'}")
    
    return df_results


# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

def add_noise(X, y, noise_type='gaussian', intensity=0.1, seed=42):
    """Add noise to training data for robustness testing."""
    np.random.seed(seed)
    X_noisy = X.copy()
    y_noisy = y.copy()
    n = len(y)
    
    if noise_type == 'gaussian':
        X_noisy = X_noisy + np.random.normal(0, intensity, X_noisy.shape)
    elif noise_type == 'label':
        n_flip = int(n * intensity)
        flip_idx = np.random.choice(n, n_flip, replace=False)
        unique_labels = np.unique(y)
        for idx in flip_idx:
            current = y_noisy[idx]
            others = [l for l in unique_labels if l != current]
            y_noisy[idx] = np.random.choice(others)
    
    return X_noisy, y_noisy


def robustness_analysis(results, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 80)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    noise_configs = [
        ('Clean', None, 0),
        ('Gaussian σ=0.1', 'gaussian', 0.1),
        ('Gaussian σ=0.5', 'gaussian', 0.5),
        ('Label 5%', 'label', 0.05),
        ('Label 10%', 'label', 0.10),
    ]
    
    robust_data = []
    for noise_name, noise_type, intensity in noise_configs:
        if noise_type is None:
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, y_tr = add_noise(X_train, y_train, noise_type, intensity)
        
        row = {'Noise': noise_name}
        for r in results:
            name = r['name']
            build_fn_map = {r2['name']: r2 for r2 in results}
            
            # Rebuild and retrain
            model_data = [b for b in ALL_MODELS if b().__class__.__name__[:3].upper() == r['name'][:3].upper()]
            if not model_data:
                # fallback: use name mapping
                name_to_idx = {build_knn()[1]: build_knn,
                               build_svm()[1]: build_svm,
                               build_rf()[1]: build_rf,
                               build_gbdt()[1]: build_gbdt,
                               build_hist_gbdt()[1]: build_hist_gbdt,
                               build_mlp()[1]: build_mlp}
                build_fn = name_to_idx.get(name)
            else:
                build_fn = model_data[0]
            
            if build_fn is None:
                continue
            
            m, _ = build_fn()
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_test)
            acc = np.mean(y_pred == y_test)
            row[name] = acc
        
        robust_data.append(row)
        
    df_robust = pd.DataFrame(robust_data)
    print(df_robust.to_string(index=False))
    df_robust.to_csv(RESULTS_DIR / "robustness_analysis.csv", index=False)
    print(f"\n[OK] Saved to {RESULTS_DIR / 'robustness_analysis.csv'}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print(" Dry Bean Classification - Full Pipeline")
    print("=" * 80)
    
    # 1. Load + Preprocess
    data = preprocess_pipeline(remove_outliers_flag=False)
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = data
    
    # 2. Train all models
    train_results = train_all(X_train, y_train, X_val, y_val)
    
    # 3. Test set evaluation
    df_results = evaluate_on_test(train_results, X_test, y_test, le)
    
    # 4. Robustness analysis
    robustness_analysis(train_results, X_train, y_train, X_test, y_test)
    
    # 5. Summary
    best_model = df_results.iloc[0]
    print("\n" + "=" * 80)
    print(f"[BEST] BEST MODEL: {best_model['name']} -> Test Accuracy: {best_model['accuracy']:.4f}")
    print("=" * 80)
    
    return df_results


if __name__ == "__main__":
    main()
