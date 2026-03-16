"""
automate_Wilda-Ariffatul-Faisalnur.py

Script otomasi preprocessing dataset Wine Quality.
Mengkonversi langkah-langkah yang ada pada notebook eksperimen
menjadi pipeline preprocessing otomatis.

Author: Wilda Ariffatul Faisalnur
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Memuat dataset Wine Quality dari file CSV.
    Dataset menggunakan separator semicolon (;).
    
    Args:
        filepath (str): Path ke file CSV dataset
    
    Returns:
        pd.DataFrame: Dataset yang sudah dimuat
    """
    print("[1/6] Memuat dataset...")
    df = pd.read_csv(filepath, sep=';')
    print(f"  Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def perform_eda(df):
    """
    Melakukan Exploratory Data Analysis sederhana.
    Menampilkan informasi dasar tentang dataset.
    
    Args:
        df (pd.DataFrame): Dataset untuk dianalisis
    """
    print("\n[2/6] Melakukan EDA...")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplikat: {df.duplicated().sum()}")
    print(f"  Distribusi quality:\n{df['quality'].value_counts().sort_index().to_string()}")
    print(f"  Statistik deskriptif:")
    print(df.describe().to_string())


def remove_duplicates(df):
    """
    Menghapus data duplikat dari dataset.
    
    Args:
        df (pd.DataFrame): Dataset input
    
    Returns:
        pd.DataFrame: Dataset tanpa duplikat
    """
    print("\n[3/6] Menghapus data duplikat...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"  Dihapus: {before - after} baris duplikat")
    print(f"  Sisa data: {after} baris")
    return df


def remove_outliers_iqr(df, columns):
    """
    Menghapus outlier menggunakan metode Interquartile Range (IQR).
    
    Args:
        df (pd.DataFrame): Dataset input
        columns (list): Daftar kolom untuk deteksi outlier
    
    Returns:
        pd.DataFrame: Dataset tanpa outlier
    """
    print("\n[4/6] Menghapus outlier (IQR)...")
    df_clean = df.copy()
    total_removed = 0
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        after = len(df_clean)
        removed = before - after
        if removed > 0:
            total_removed += removed
            print(f"  {col}: {removed} outlier dihapus")
    
    print(f"  Total outlier dihapus: {total_removed}")
    print(f"  Sisa data: {len(df_clean)} baris")
    return df_clean


def preprocess_data(df):
    """
    Melakukan preprocessing data:
    - Konversi quality ke binary (good/bad)
    - StandardScaler normalisasi
    - Train-test split (80:20)
    
    Args:
        df (pd.DataFrame): Dataset yang sudah dibersihkan
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) hasil preprocessing
    """
    print("\n[5/6] Preprocessing data...")
    
    # Binning: quality >= 7 -> Good (1), < 7 -> Bad (0)
    df['quality_label'] = (df['quality'] >= 7).astype(int)
    print(f"  Distribusi quality_label (0=Bad, 1=Good):")
    print(f"  {df['quality_label'].value_counts().to_string()}")
    
    # Hapus kolom quality asli
    df = df.drop('quality', axis=1)
    
    # Split fitur dan target
    X = df.drop('quality_label', axis=1)
    y = df['quality_label']
    
    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Menyimpan dataset hasil preprocessing ke file CSV.
    
    Args:
        X_train, X_test, y_train, y_test: Data hasil split
        output_dir (str): Direktori output
    """
    print("\n[6/6] Menyimpan dataset hasil preprocessing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Gabungkan X dan y, lalu simpan
    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    train_path = os.path.join(output_dir, 'winequality_preprocessing_train.csv')
    test_path = os.path.join(output_dir, 'winequality_preprocessing_test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"  Train data disimpan: {train_path} ({train_data.shape})")
    print(f"  Test data disimpan: {test_path} ({test_data.shape})")
    print("\n✅ Preprocessing selesai!")


def main():
    """
    Fungsi utama yang mengorkestrasi seluruh pipeline preprocessing.
    """
    print("=" * 60)
    print("AUTOMATED PREPROCESSING - WINE QUALITY DATASET")
    print("Author: Wilda Ariffatul Faisalnur")
    print("=" * 60)
    
    # Tentukan path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, '..', 'winequality-red.csv')
    output_dir = os.path.join(script_dir, 'namadataset_preprocessing')
    
    # Jika raw data tidak ditemukan di parent, coba di direktori saat ini
    if not os.path.exists(raw_data_path):
        raw_data_path = os.path.join(script_dir, 'winequality-red.csv')
    if not os.path.exists(raw_data_path):
        raw_data_path = 'winequality-red.csv'
    
    # 1. Load data
    df = load_data(raw_data_path)
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Hapus duplikat
    df = remove_duplicates(df)
    
    # 4. Hapus outlier
    feature_cols = df.columns.drop('quality')
    df = remove_outliers_iqr(df, feature_cols)
    
    # 5. Preprocessing (binning, scaling, split)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 6. Simpan hasil
    save_data(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
