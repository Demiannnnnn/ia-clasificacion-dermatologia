from src.training.trainer import train_model
from src.training.evaluator import evaluate_model
from src.visualization.plotter import plot_confusion_matrix
from src.visualization.report_generator import generate_report

def main():
    """
    Main function to run the machine learning pipeline.
    """
    # Define paths
    data_path = 'dermatology/dermatology.data'
    report_path = 'evaluation_report.txt'
    cm_plot_path = 'confusion_matrix.png'

    # Class names for plotting
    class_names = [
        '1: psoriasis', '2: seboreic dermatitis', '3: lichen planus',
        '4: pityriasis rosea', '5: cronic dermatitis', '6: pityriasis rubra pilaris'
    ]

    # Train the model
    print("Training model...")
    model, X_test, y_test = train_model(data_path)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    print("Model evaluation complete.")

    # Generate report
    print(f"Generating report at {report_path}...")
    generate_report(results, report_path)
    print("Report generation complete.")

    # Plot confusion matrix
    print(f"Generating confusion matrix plot at {cm_plot_path}...")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
    print("Plot generation complete.")

    print("\nPipeline finished successfully!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"See {report_path} and {cm_plot_path} for detailed results.")


if __name__ == '__main__':
    main()
