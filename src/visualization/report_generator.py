def generate_report(results, report_path):
    """
    Generates a text report of the evaluation results.

    Args:
        results (dict): A dictionary containing the evaluation metrics.
        report_path (str): The path to save the report.
    """
    with open(report_path, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("===================\n\n")
        f.write(f"Accuracy: {results['accuracy']}\n\n")
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(results['confusion_matrix']))

if __name__ == '__main__':
    # This is an example of how to use the function
    # We need to train and evaluate a model first
    from src.training.trainer import train_model
    from src.training.evaluator import evaluate_model

    data_path = '../../dermatology/dermatology.data'
    trained_model, X_test, y_test = train_model(data_path)
    results = evaluate_model(trained_model, X_test, y_test)

    generate_report(results, 'evaluation_report.txt')
    print("Evaluation report saved to evaluation_report.txt")
