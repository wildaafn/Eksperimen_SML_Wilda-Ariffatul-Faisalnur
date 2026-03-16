import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def automate_preprocessing(file_path):
    print(f"🚀 Mulai proses data dari: {file_path}")
    
    # [1] Baca data
    # Dataset Rice Cammeo & Osmancik dari UCI
    df = pd.read_csv(file_path)
    print(f"Data awal: {df.shape[0]} baris, {df.shape[1]} kolom")

    # [2] EDA Singkat (Penting buat dokumentasi)
    print("Bikin plot korelasi fitur...")
    plt.figure(figsize=(10, 8))
    # Drop target biar heatmap isinya fitur doang
    numeric_df = df.drop(columns=['Class'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasi Fitur Morfologi Beras')
    plt.tight_layout()
    plt.savefig("/Volumes/Data/Program/ML PROJECT/preprocessing/rice_correlation.png")
    
    # [3] Bersih-bersih (Handle Missing Values)
    # Cek kalau ada yang kosong, langsung buang aja barisnya
    if df.isnull().values.any():
        print("Ditemukan missing values, membersihkan...")
        df = df.dropna()

    # [4] Label Encoding buat target 'Class'
    # Cammeo -> 0, Osmancik -> 1
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])
    print(f"Target diubah jadi angka: {list(le.classes_)} -> [0, 1]")

    # [5] Buang Outlier (Metode IQR)
    # Biar model lebih stabil, data extrem kita buang
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Sisa data setelah outlier dibuang: {df.shape[0]} baris")

    # [6] Split Data (Standar 80% Train, 20% Test)
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # [7] Scaling data biar RandomForest kerjanya enak
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Kembalikan ke format CSV (simpan fitur + target)
    train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_final['Class'] = y_train.values
    
    test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_final['Class'] = y_test.values

    # [8] Simpan file buat dipake step training nanti
    output_dir = "/Volumes/Data/Program/ML PROJECT/preprocessing/namadataset_preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    
    train_final.to_csv(f"{output_dir}/rice_preprocessing_train.csv", index=False)
    test_final.to_csv(f"{output_dir}/rice_preprocessing_test.csv", index=False)
    
    print(f"✅ Preprocessing BERES! File tersimpan di '{output_dir}'")

if __name__ == "__main__":
    automate_preprocessing("/Volumes/Data/Program/ML PROJECT/rice_dataset.csv")
