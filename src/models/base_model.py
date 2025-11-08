from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.

        Args:
            X_train: The training features.
            y_train: The training labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Makes predictions on the test set.

        Args:
            X_test: The test features.

        Returns:
            The predicted labels.
        """
        pass
