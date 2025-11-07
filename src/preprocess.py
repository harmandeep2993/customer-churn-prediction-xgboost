import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def drop_columns(df, columns):
    return df.drop(columns, axis=1)

def encode_binary_columns(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

def convert_total_charges(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

def scale_numerical(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
