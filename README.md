# IA-Dermatology

[![CI](https://github.com/Demiannnnnn/ia-clasificacion-dermatologia/actions/workflows/ci.yml/badge.svg?branch=tests/unitarios-y-ci)](https://github.com/Demiannnnnn/ia-clasificacion-dermatologia/actions/workflows/ci.yml)

Machine Learning project for automated classification of dermatological diseases using Naive Bayes algorithm. Trained on UCI Dermatology dataset with 366 instances and 34 clinical features across 6 disease classes.

## Dataset

UCI Dermatology Database (Ilter & Guvenir, 1998), included in `dermatology/`:
`dermatology.data` (the data) and `dermatology.names` (attribute and class description).
Missing `age` values (`?`) are filled with the column mean during cleaning.

## Stack

Python 3 · pandas · NumPy · scikit-learn (`GaussianNB`, `StandardScaler`) · matplotlib · seaborn · pytest

## Project layout

```
main.py                 # Runs the full pipeline
src/data/               # Loading and exploratory analysis
src/evaluation/         # Cleaning, stratified train/test split, scaling
src/models/             # Naive Bayes model wrapper
src/training/           # Training, evaluation and metrics
src/visualization/      # Plots and text report
tests/                  # pytest suite
reports/                # Generated plots and evaluation report
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

The pipeline splits the data 80/20 (stratified by class, `random_state=42`), trains the model and writes
`reports/evaluation_report.txt`, `reports/confusion_matrix.png` and the exploratory plots.
The current report gives an accuracy of 0.865 on the 74-sample test set.

## Tests

```bash
pip install -r requirements.txt pytest
MPLBACKEND=Agg python -m pytest -v
```

The same suite runs on GitHub Actions on every push and pull request.

## License

Apache License 2.0 (see `LICENSE`).
