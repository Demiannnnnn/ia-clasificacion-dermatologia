from src.data.loader import load_data
from src.evaluation.cleaner import clean_data
from src.evaluation.splitter import split_data
from src.evaluation.scaler import scale_features
from src.models.naive_bayes import NaiveBayesModel

def train_model(data_path):
    """
    Trains a Naive Bayes model on the dermatology dataset.

    Args:
        data_path (str): The path to the dermatology.data file.

    Returns:
        tuple: A tuple containing the trained model, X_test_scaled, and y_test.
    """
    # Load data
    dermatology_df = load_data(data_path)

    # Clean data
    dermatology_df = clean_data(dermatology_df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(dermatology_df)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Initialize and train the model
    model = NaiveBayesModel()
    model.train(X_train_scaled, y_train)

    return model, X_test_scaled, y_test

if __name__ == '__main__':
    # This is an example of how to use the function
    data_path = '../../dermatology/dermatology.data'
    trained_model, X_test, y_test = train_model(data_path)

    print("Model trained:", trained_model is not None)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
