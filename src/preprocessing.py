"""Preprocessing pipeline for the credit risk dataset.

Extracted from notebooks/credit_risk_modeling_en.ipynb so the logic can be
unit-tested and reused independently of the notebook.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
ONE_HOT_COLUMNS = ['person_home_ownership', 'loan_intent']
TARGET_COLUMN = 'loan_status'


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add missingness indicators, drop unrealistic outliers, impute medians."""
    df = df.copy()

    df['is_missing_rate'] = df['loan_int_rate'].isna().astype(int)
    df['is_missing_emplength'] = df['person_emp_length'].isna().astype(int)

    df = df[
        (df['person_age'] < 100)
        & ((df['person_emp_length'] < 60) | (df['person_emp_length'].isna()))
    ].copy()

    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary, ordinal, and nominal categorical variables."""
    df = df.copy()

    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
    df['loan_grade'] = df['loan_grade'].map(GRADE_MAP)
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True)

    return df


def split_features_target(df: pd.DataFrame, random_state: int, test_size: float = 0.2):
    """Stratified train/test split preserving the default ratio."""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
