import numpy as np
import pandas as pd

from src.evaluation.scaler import scale_features
from src.evaluation.splitter import split_data
from src.models.naive_bayes import NaiveBayesModel
from src.training.evaluator import evaluate_model
from src.training.metrics import get_accuracy, get_confusion_matrix
from src.training.trainer import train_model
from src.visualization.plotter import plot_confusion_matrix
from src.visualization.report_generator import generate_report
from tests.conftest import DATA_PATH


def _toy_df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "class": [1, 2] * (n // 2),
    })


def test_split_data_sizes_and_reproducibility():
    df = _toy_df()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    assert len(X_train) == 40 and len(X_test) == 10
    assert "class" not in X_train.columns
    again = split_data(df, test_size=0.2)
    assert list(X_test.index) == list(again[1].index)


def test_scale_features_fits_only_on_train():
    X_train = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
    X_test = pd.DataFrame({"x": [2.0, 6.0]})
    tr, te = scale_features(X_train, X_test)
    assert np.allclose(tr.mean(axis=0), 0.0)
    assert np.allclose(tr.std(axis=0), 1.0)
    # mean 2, std sqrt(8/3) from train
    assert np.allclose(te[:, 0], [0.0, 4.0 / np.sqrt(8.0 / 3.0)])


def test_naive_bayes_separable_data():
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    model = NaiveBayesModel()
    model.train(X, y)
    assert list(model.predict(np.array([[0.05], [5.05]]))) == [1, 2]


def test_metrics():
    y_true = [1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 1, 3]
    assert get_accuracy(y_true, y_pred) == 5 / 6
    cm = get_confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[2, 0, 0], [1, 1, 0], [0, 0, 2]]


def test_end_to_end_training_and_reports(tmp_path):
    model, X_test, y_test = train_model(DATA_PATH)
    results = evaluate_model(model, X_test, y_test)
    assert X_test.shape == (74, 34)
    assert results["accuracy"] > 0.8
    assert results["confusion_matrix"].sum() == len(y_test)

    report = tmp_path / "report.txt"
    generate_report(results, str(report))
    text = report.read_text()
    assert "Accuracy:" in text and "Confusion Matrix:" in text

    png = tmp_path / "cm.png"
    plot_confusion_matrix(results["confusion_matrix"], [str(i) for i in range(1, 7)], str(png))
    assert png.stat().st_size > 0
