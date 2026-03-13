import pandas as pd

def get_categorical_columns (data:pd.DataFrame)->list[str]:
    """Return a list of column names that are categorical."""
    cat_cols = data.select_dtypes(include='object').columns
    return cat_cols

def get_numerical_columns(data:pd.DataFrame)->list[str]:
    """Return a list of column names that are numerical."""
    num_cols = data.select_dtypes(exclude='object').columns
    return num_cols

def get_X_and_y (data:pd.DataFrame, target_column:str)->tuple[pd.DataFrame]:
    X = data.drop(columns=[target_column],axis=1)
    Y = data[target_column]
    return X, Y