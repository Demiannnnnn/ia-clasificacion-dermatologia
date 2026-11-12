import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_data
from src.evaluation.cleaner import clean_data
from tests.conftest import DATA_PATH


@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


def test_loader_shape_and_columns(raw_df):
    assert raw_df.shape == (366, 35)
    assert raw_df.columns[-1] == "class"
    assert raw_df.columns[-2] == "age"


def test_loader_marks_question_marks_as_nan(raw_df):
    # The UCI dataset has 8 missing ages encoded as '?'
    assert raw_df["age"].isna().sum() == 8


def test_loader_classes_are_1_to_6(raw_df):
    assert sorted(raw_df["class"].unique()) == [1, 2, 3, 4, 5, 6]


def test_clean_data_fills_age_with_mean():
    df = pd.DataFrame({"a": [1, 2, 3], "age": [10.0, np.nan, 30.0], "class": [1, 2, 1]})
    cleaned = clean_data(df.copy())
    assert cleaned["age"].tolist() == [10.0, 20.0, 30.0]
    assert len(cleaned) == 3


def test_clean_data_coerces_and_drops_non_numeric_rows():
    df = pd.DataFrame({"a": ["1", "x", "3"], "age": [10, 20, 30], "class": [1, 2, 1]})
    cleaned = clean_data(df.copy())
    assert len(cleaned) == 2
    assert cleaned["a"].tolist() == [1.0, 3.0]


def test_clean_real_dataset_has_no_nans(raw_df):
    cleaned = clean_data(raw_df.copy())
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == 366


def test_clean_data_does_not_mutate_input():
    df = pd.DataFrame({"a": ["1", "2", "3"], "age": ["10", "?", "30"], "class": [1, 2, 1]})
    snapshot = df.copy(deep=True)
    cleaned = clean_data(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert not cleaned.equals(df)

