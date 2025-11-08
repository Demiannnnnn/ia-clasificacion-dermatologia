from src.models.base_model import BaseModel
from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel(BaseModel):
    """
    A Naive Bayes model implementation.
    """
    def __init__(self):
        self.model = GaussianNB()

    def train(self, X_train, y_train):
        """
        Trains the Naive Bayes model.

        Args:
            X_train: The training features.
            y_train: The training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the trained Naive Bayes model.

        Args:
            X_test: The test features.

        Returns:
            The predicted labels.
        """
        return self.model.predict(X_test)

if __name__ == '__main__':
    # This is an example of how to use the class
    # We need to load, split and scale the data first
    from src.data.loader import load_data
    from src.evaluation.splitter import split_data
    from src.evaluation.scaler import scale_features

    data_path = '../../dermatology/dermatology.data'
    dermatology_df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(dermatology_df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Initialize and train the model
    nb_model = NaiveBayesModel()
    nb_model.train(X_train_scaled, y_train)

    # Make predictions
    predictions = nb_model.predict(X_test_scaled)

    print("Predictions:", predictions)
    print("Actual labels:", y_test.values)
