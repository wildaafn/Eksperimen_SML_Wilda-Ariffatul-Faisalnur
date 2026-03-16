import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Konfigurasi Path
DATA_PATH = "/Volumes/Data/Program/ML PROJECT/rice_dataset.csv"
OUTPUT_DIR = "/Volumes/Data/Program/ML PROJECT/preprocessing/namadataset_preprocessing"
SUBMISSION_DIR = "/Volumes/Data/Program/ML PROJECT/SMSML_Wilda-Ariffatul-Faisalnur/preprocessing/namadataset_preprocessing"

def load_data(path):
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def perform_eda(df):
    print("📊 Performing EDA...")
    print(f"Info Dataset:\n{df.info()}")
    print(f"Statistik Deskriptif:\n{df.describe()}")
    
    # Distribusi Target
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Class', palette='viridis')
    plt.title('Distribusi Jenis Beras (Cammeo vs Osmancik)')
    plt.savefig("/Volumes/Data/Program/ML PROJECT/preprocessing/rice_distribution.png")
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.drop(columns=['Class'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Korelasi Fitur Morfologi Beras')
    plt.savefig("/Volumes/Data/Program/ML PROJECT/preprocessing/rice_correlation.png")
    print("✅ EDA plots saved.")

def preprocess_data(df):
    print("⚙️ Preprocessing data...")
    
    # 1. Handling Missing Values
    df = df.dropna()
    
    # 2. Label Encoding Target (Cammeo=0, Osmancik=1)
    df['Class'] = df['Class'].map({'Cammeo': 0, 'Osmancik': 1})
    
    # 3. Outlier Removal (IQR Method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Final check for NaNs after filtering
    df = df.dropna()
    
    # 4. Feature Scaling (StandardScaler)
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    
    # Gabungkan kembali untuk disimpan
    train_data = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    return train_data, test_data

def main():
    # Buat direktori jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    
    # Load
    df = load_data(DATA_PATH)
    
    # EDA
    perform_eda(df)
    
    # Preprocess
    train_df, test_df = preprocess_data(df)
    
    # Save Output
    train_path = os.path.join(OUTPUT_DIR, "rice_preprocessing_train.csv")
    test_path = os.path.join(OUTPUT_DIR, "rice_preprocessing_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Copy to submission folder
    train_sub_path = os.path.join(SUBMISSION_DIR, "rice_preprocessing_train.csv")
    test_sub_path = os.path.join(SUBMISSION_DIR, "rice_preprocessing_test.csv")
    train_df.to_csv(train_sub_path, index=False)
    test_df.to_csv(test_sub_path, index=False)
    
    print(f"✅ Preprocessing Selesai!")
    print(f"Data Train: {train_df.shape}")
    print(f"Data Test : {test_df.shape}")
    print(f"File disimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
