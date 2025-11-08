import os
from src.data.explorer import explore_data
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
    reports_dir = 'reports'
    report_path = os.path.join(reports_dir, 'evaluation_report.txt')
    cm_plot_path = os.path.join(reports_dir, 'confusion_matrix.png')

    # Create reports directory if it doesn't exist
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # --- 1. Exploratory Data Analysis ---
    explore_data(data_path, reports_dir)

    # Class names for plotting
    class_names = [
        '1: psoriasis', '2: seboreic dermatitis', '3: lichen planus',
        '4: pityriasis rosea', '5: cronic dermatitis', '6: pityriasis rubra pilaris'
    ]

    # --- 2. Model Training ---
    print("\n--- Model Training ---")
    model, X_test, y_test = train_model(data_path)
    print("Model training complete.")

    # --- 3. Model Evaluation ---
    print("\n--- Model Evaluation ---")
    results = evaluate_model(model, X_test, y_test)
    print("Model evaluation complete.")

    # --- 4. Report Generation ---
    print("\n--- Report Generation ---")
    print(f"Generating report at {report_path}...")
    generate_report(results, report_path)
    print("Report generation complete.")

    # Plot confusion matrix
    print(f"Generating confusion matrix plot at {cm_plot_path}...")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_plot_path)
    print("Plot generation complete.")

    print("\nPipeline finished successfully!")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"See the '{reports_dir}/' directory for detailed reports and plots.")


if __name__ == '__main__':
    main()
