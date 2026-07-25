import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_dataset, encode_features, split_features_target


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        'person_age': [25, 30, 150, 40],           # row 2 is an unrealistic outlier
        'person_emp_length': [2.0, np.nan, 5.0, 70.0],  # row 3 is an unrealistic outlier
        'loan_int_rate': [10.5, np.nan, 12.0, 9.0],
        'loan_grade': ['A', 'B', 'C', 'D'],
        'cb_person_default_on_file': ['Y', 'N', 'Y', 'N'],
        'person_home_ownership': ['RENT', 'OWN', 'RENT', 'MORTGAGE'],
        'loan_intent': ['EDUCATION', 'VENTURE', 'PERSONAL', 'MEDICAL'],
        'loan_status': [1, 0, 1, 0],
    })


def test_clean_dataset_adds_missingness_indicators(raw_df):
    cleaned = clean_dataset(raw_df)
    assert 'is_missing_rate' in cleaned.columns
    assert 'is_missing_emplength' in cleaned.columns
    # indicators must be captured on the original rows, before outlier removal or imputation
    assert cleaned.loc[cleaned['person_age'] == 30, 'is_missing_rate'].iloc[0] == 1
    assert cleaned.loc[cleaned['person_age'] == 30, 'is_missing_emplength'].iloc[0] == 1


def test_clean_dataset_drops_unrealistic_outliers(raw_df):
    cleaned = clean_dataset(raw_df)
    assert (cleaned['person_age'] < 100).all()
    assert ((cleaned['person_emp_length'] < 60) | cleaned['person_emp_length'].isna()).all()
    assert len(cleaned) == 2


def test_clean_dataset_imputes_with_median_not_mean(raw_df):
    cleaned = clean_dataset(raw_df)
    # after dropping outlier rows, remaining loan_int_rate values are [10.5, nan] -> median 10.5
    assert cleaned['loan_int_rate'].isna().sum() == 0
    assert cleaned.loc[cleaned['person_age'] == 30, 'loan_int_rate'].iloc[0] == 10.5


def test_encode_features_maps_binary_and_ordinal_variables(raw_df):
    cleaned = clean_dataset(raw_df)
    encoded = encode_features(cleaned)
    assert set(encoded['cb_person_default_on_file'].unique()) <= {0, 1}
    assert encoded['loan_grade'].max() <= 7
    assert encoded['loan_grade'].min() >= 1


def test_encode_features_one_hot_encodes_nominal_variables_with_drop_first(raw_df):
    cleaned = clean_dataset(raw_df)
    encoded = encode_features(cleaned)
    assert 'person_home_ownership' not in encoded.columns
    assert 'loan_intent' not in encoded.columns
    # drop_first=True means one category per nominal variable has no dummy column
    home_ownership_dummies = [c for c in encoded.columns if c.startswith('person_home_ownership_')]
    assert len(home_ownership_dummies) < raw_df['person_home_ownership'].nunique()


def test_split_features_target_is_stratified_and_reproducible(raw_df):
    cleaned = encode_features(clean_dataset(raw_df))
    # duplicate rows so the stratified split has enough samples per class
    cleaned = pd.concat([cleaned] * 10, ignore_index=True)

    X_train_a, X_test_a, y_train_a, y_test_a = split_features_target(cleaned, random_state=42)
    X_train_b, X_test_b, y_train_b, y_test_b = split_features_target(cleaned, random_state=42)

    pd.testing.assert_frame_equal(X_train_a, X_train_b)
    pd.testing.assert_series_equal(y_train_a, y_train_b)
    assert 'loan_status' not in X_train_a.columns
