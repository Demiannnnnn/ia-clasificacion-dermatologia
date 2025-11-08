from src.training.metrics import get_accuracy, get_classification_report, get_confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained model.

    Args:
        model: The trained model.
        X_test: The test features.
        y_test: The test labels.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = get_accuracy(y_test, y_pred)
    report = get_classification_report(y_test, y_pred)
    cm = get_confusion_matrix(y_test, y_pred)

    evaluation_results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

    return evaluation_results

if __name__ == '__main__':
    # This is an example of how to use the function
    # We need to train a model first
    from src.training.trainer import train_model

    data_path = '../../dermatology/dermatology.data'
    trained_model, X_test, y_test = train_model(data_path)

    # Evaluate the model
    results = evaluate_model(trained_model, X_test, y_test)

    print("Accuracy:", results['accuracy'])
    print("Classification Report:\n", results['classification_report'])
    print("Confusion Matrix:\n", results['confusion_matrix'])
