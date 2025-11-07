import pandas as pd
from src.preprocess import (
    strip_whitespace,
    convert_total_charges,
    unify_service_columns,
    encode_binary_columns,
    drop_identifier,
    impute_missing,
    one_hot_encode,
    scale_numerical
)

def full_preprocess(df):
    """
    Run all preprocessing steps on raw input DataFrame.
    
    Parameters:
        df (pd.DataFrame): Raw input data
    
    Returns:
        pd.DataFrame: Cleaned and transformed data
    """
    df = strip_whitespace(df)
    df = convert_total_charges(df)
    df = unify_service_columns(df)
    df = encode_binary_columns(df)
    df = drop_identifier(df)
    df = impute_missing(df)

    # One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    df = one_hot_encode(df, cat_cols)

    # Scale numerical columns (excluding target)
    num_cols = df.select_dtypes(include=["int64", "float64"]) \
                 .drop(columns=["Churn"], errors="ignore") \
                 .columns.tolist()
    df = scale_numerical(df, num_cols)

    return df