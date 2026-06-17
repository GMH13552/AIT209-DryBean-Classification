"""
Quick training run for Dry Bean Classification
"""
import numpy as np
import pandas as pd
import time
import sys
from preprocess import preprocess_pipeline

# Load data
print("Loading & preprocessing...")
sys.stdout.flush()
X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = preprocess_pipeline()

print(f"\nData ready: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
sys.stdout.flush()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

models = [
    ("KNN               ", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ("SVM (RBF)         ", SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
    ("Random Forest     ", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
    ("Gradient Boosting ", GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, max_features=0.8, random_state=42)),
    ("HistGrad Boosting ", HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42)),
    ("MLP               ", MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=50, early_stopping=True, random_state=42)),
]

results = []
for name, model in models:
    print(f"\nTraining {name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Inference speed
    t1 = time.perf_counter()
    for _ in range(100):
        model.predict(X_test)
    speed = (time.perf_counter() - t1) / 100 * 1000
    
    print(f"Done ({train_time:.1f}s)", flush=True)
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | F1: {test_f1:.4f} | Speed: {speed:.1f}ms")
    
    results.append({
        'Model': name.strip(),
        'Train Acc': train_acc,
        'Val Acc': val_acc,
        'Test Acc': test_acc,
        'Test F1': test_f1,
        'Overfit Gap': train_acc - val_acc,
        'Train Time (s)': train_time,
        'Inference (ms)': speed
    })

# Final table
df = pd.DataFrame(results).sort_values('Test Acc', ascending=False)
print("\n" + "=" * 90)
print("FINAL RESULTS")
print("=" * 90)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv("results/model_comparison.csv", index=False)
print(f"\nBest: {df.iloc[0]['Model']} with Test Acc = {df.iloc[0]['Test Acc']:.4f}")
