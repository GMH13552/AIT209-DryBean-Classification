"""
Dry Bean Classification - Data Loader
Loads train/val/test CSVs, provides basic info.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    """Load all three datasets."""
    train = pd.read_csv(DATA_DIR / "Dry_Bean_Dataset_Dirty_train.csv")
    val = pd.read_csv(DATA_DIR / "Dry_Bean_Dataset_Dirty_val.csv")
    test = pd.read_csv(DATA_DIR / "Dry_Bean_Dataset_Dirty_test.csv")
    return train, val, test


def get_class_mapping():
    """Map dirty class labels to canonical 7 classes."""
    mapping = {}
    # BARBUNYA
    for v in ['barbunya', 'BARBUNYA', 'BARBUNYA ']:
        mapping[v] = 'BARBUNYA'
    # BOMBAY
    for v in ['bombay', 'BOMBAY', 'B0MBAY', 'BOMBAY ']:
        mapping[v] = 'BOMBAY'
    # CALI
    for v in ['cali', 'CALI', 'CALI ']:
        mapping[v] = 'CALI'
    # DERMASON
    for v in ['dermason', 'DERMASON', 'D3RMAS0N', 'DERMASON ', 'DERMASON  ']:
        mapping[v] = 'DERMASON'
    # HOROZ
    for v in ['horoz', 'HOROZ', 'H0R0Z', 'HOROZ ']:
        mapping[v] = 'HOROZ'
    # SEKER
    for v in ['seker', 'SEKER', 'S3K3R', 'SEKER ']:
        mapping[v] = 'SEKER'
    # SIRA
    for v in ['sira', 'SIRA', 'SIRA ']:
        mapping[v] = 'SIRA'
    return mapping


if __name__ == "__main__":
    train, val, test = load_data()
    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")
    print(f"\nTrain classes:\n{train['Class'].value_counts()}")
