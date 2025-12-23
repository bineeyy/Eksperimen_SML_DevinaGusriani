import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump

def preprocess_data(data, target_column, save_pipeline_path, output_path):
    # =====================
    # DROP KOLOM TIDAK RELEVAN
    # =====================
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])

    # =====================
    # HANDLE MISSING & DUPLICATE
    # =====================
    data = data.dropna()
    data = data.drop_duplicates()

    # =====================
    # OUTLIER HANDLING (IQR CLIPPING)
    # =====================
    num_cols = ['Age', 'Height', 'Weight', 'BMI']
    for col in num_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower, upper)

    # =====================
    # AGE BINNING
    # =====================
    data['Age_bin'] = pd.cut(
        data['Age'],
        bins=[10, 20, 59, 120],
        labels=[0, 1, 3],
        include_lowest=True
    )

    # =====================
    # ENCODING
    # =====================
    le_gender = LabelEncoder()
    data['Gender'] = le_gender.fit_transform(data['Gender'])

    le_label = LabelEncoder()
    data[target_column] = le_label.fit_transform(data[target_column])

    # =====================
    # PIPELINE (SCALING ONLY)
    # =====================
    feature_cols = data.columns.drop(target_column)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, feature_cols)
        ]
    )

    # Fit & transform ALL data (tanpa splitting)
    X_processed = preprocessor.fit_transform(data[feature_cols])

    # =====================
    # SAVE OUTPUT
    # =====================
    os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
    dump(preprocessor, save_pipeline_path)

    # BUAT FOLDER OUTPUT (INI FIX-NYA)
    os.makedirs(output_path, exist_ok=True)

    processed_df = pd.DataFrame(
        X_processed,
        columns=feature_cols
    )
    processed_df[target_column] = data[target_column].values

    processed_df.to_csv(
        os.path.join(output_path, "obesity_clean.csv"),
        index=False
    )


    print("Preprocessing selesai. Dataset siap training disimpan.")

    return processed_df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Devina\Documents\SEMESTER 5\Eksperimen_SML_Devina\obesity_raw\Obesity Classification.csv")

    preprocess_data(
        data=df,
        target_column="Label",
        save_pipeline_path="pipeline/preprocessor.joblib",
        output_path="preprocessing/obesity_preprocessing"
    )
