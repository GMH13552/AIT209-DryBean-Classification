"""
Dry Bean Classification - Data Preprocessing & Feature Engineering
Handles all data quality issues identified in EDA:
1. Class label normalization (case + typo fixes)
2. Solidity: "?" → NaN → median imputation
3. Compactness: strip " cm" → float
4. Perimeter: NaN → median imputation
5. StandardScaler normalization
6. Optional: outlier handling, PCA
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from data_loader import load_data, get_class_mapping


def clean_labels(df):
    """Normalize class labels to canonical 7 classes."""
    mapping = get_class_mapping()
    df['Class'] = df['Class'].str.strip().map(
        lambda x: mapping.get(x, x.upper())
    )
    return df


def fix_solidity(df):
    """Replace '?' with NaN, then convert to float."""
    df['Solidity'] = pd.to_numeric(df['Solidity'], errors='coerce')
    return df


def fix_compactness(df):
    """Strip ' cm' suffix and convert to float."""
    df['Compactness'] = (
        df['Compactness']
        .astype(str)
        .str.replace(' cm', '', regex=False)
        .str.strip()
    )
    df['Compactness'] = pd.to_numeric(df['Compactness'], errors='coerce')
    return df


def impute_missing(df, imputer=None, fit=True):
    """Impute missing values using median strategy."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude label columns
    numeric_cols = [c for c in numeric_cols if c not in ('Class', 'label')]
    
    if fit:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df, imputer
    else:
        df[numeric_cols] = imputer.transform(df[numeric_cols])
        return df


def remove_outliers(df, contamination=0.01):
    """Optional: remove rows with extreme outlier values using IsolationForest."""
    from sklearn.ensemble import IsolationForest
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ('Class', 'label')]
    
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outliers = iso.fit_predict(df[numeric_cols])
    df_clean = df[outliers == 1].copy()
    print(f"  Outlier removal: {df.shape[0]} → {df_clean.shape[0]} "
          f"({(df.shape[0]-df_clean.shape[0])/df.shape[0]*100:.1f}% removed)")
    return df_clean


def preprocess_pipeline(save_encoders=False, remove_outliers_flag=False):
    """
    Complete preprocessing pipeline.
    
    Returns:
        X_train, X_val, X_test (scaled arrays)
        y_train, y_val, y_test (encoded labels)
        le (LabelEncoder)
        scaler (StandardScaler)
        feature_names (list)
    """
    train, val, test = load_data()
    print(f"原始数据: train={train.shape}, val={val.shape}, test={test.shape}")
    
    # Step 1: Clean labels
    train = clean_labels(train)
    val = clean_labels(val)
    test = clean_labels(test)
    
    # Step 2: Fix Solidity (? → NaN)
    train = fix_solidity(train)
    val = fix_solidity(val)
    test = fix_solidity(test)
    
    # Step 3: Fix Compactness (strip " cm")
    train = fix_compactness(train)
    val = fix_compactness(val)
    test = fix_compactness(test)
    
    # Step 4: Encode labels BEFORE outlier removal (to keep consistency)
    le = LabelEncoder()
    y_train = le.fit_transform(train['Class'])
    y_val = le.transform(val['Class'])
    y_test = le.transform(test['Class'])
    
    # Step 5: Drop Class column, keep features
    feature_cols = [c for c in train.columns if c != 'Class']
    X_train = train[feature_cols].copy()
    X_val = val[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Step 6: Impute missing values
    X_train, imputer = impute_missing(X_train, fit=True)
    X_val = impute_missing(X_val, imputer=imputer, fit=False)
    X_test = impute_missing(X_test, imputer=imputer, fit=False)
    
    # Step 7 (Optional): Remove outliers
    if remove_outliers_flag:
        train_combined = X_train.copy()
        train_combined['label'] = y_train
        train_combined = remove_outliers(train_combined)
        y_train = train_combined['label'].values
        X_train = train_combined.drop(columns=['label'])
    
    # Step 8: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"预处理完成: X_train={X_train_scaled.shape}, X_val={X_val_scaled.shape}, "
          f"X_test={X_test_scaled.shape}")
    print(f"类别: {le.classes_.tolist()}")
    print(f"类别分布: {dict(zip(le.classes_, np.bincount(y_train)))}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test, le, scaler, feature_cols)


if __name__ == "__main__":
    data = preprocess_pipeline()
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = data
    print(f"\nFeatures ({len(features)}): {features}")
