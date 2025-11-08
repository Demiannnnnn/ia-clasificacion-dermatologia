import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plots the confusion matrix and saves it to a file.

    Args:
        cm (numpy.ndarray): The confusion matrix.
        class_names (list): The names of the classes.
        output_path (str): The path to save the plot.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # This is an example of how to use the function
    # We need to train and evaluate a model first
    from src.training.trainer import train_model
    from src.training.evaluator import evaluate_model

    data_path = '../../dermatology/dermatology.data'
    trained_model, X_test, y_test = train_model(data_path)
    results = evaluate_model(trained_model, X_test, y_test)
    cm = results['confusion_matrix']
    
    class_names = ['psoriasis', 'seboreic dermatitis', 'lichen planus', 
                   'pityriasis rosea', 'cronic dermatitis', 'pityriasis rubra pilaris']

    plot_confusion_matrix(cm, class_names, 'confusion_matrix.png')
    print("Confusion matrix plot saved to confusion_matrix.png")
