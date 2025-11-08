from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_accuracy(y_true, y_pred):
    """
    Calculates the accuracy score.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        float: The accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def get_classification_report(y_true, y_pred):
    """
    Generates a classification report.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        str: The classification report.
    """
    return classification_report(y_true, y_pred)

def get_confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        numpy.ndarray: The confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    # This is an example of how to use the functions
    y_true = [1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 1, 3]

    accuracy = get_accuracy(y_true, y_pred)
    report = get_classification_report(y_true, y_pred)
    cm = get_confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
