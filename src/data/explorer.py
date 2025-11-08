import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.data.loader import load_data
from src.evaluation.cleaner import clean_data

def explore_data(data_path, output_dir):
    """
    Performs exploratory data analysis on the dataset.

    Args:
        data_path (str): Path to the data file.
        output_dir (str): Directory to save the plots.
    """
    # Load and clean the data for accurate exploration
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df.copy())

    print("--- Exploratory Data Analysis ---")
    print("\nDataset Info (after cleaning):")
    cleaned_df.info()

    print("\nDataset Description:")
    print(cleaned_df.describe())

    # --- Class Distribution ---
    plt.figure(figsize=(12, 6))
    sns.countplot(x='class', data=cleaned_df)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    class_dist_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(class_dist_path)
    plt.close()
    print(f"\nSaved class distribution plot to {class_dist_path}")

    # --- Age Distribution ---
    plt.figure(figsize=(10, 6))
    sns.histplot(cleaned_df['age'], kde=True, bins=20)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    age_dist_path = os.path.join(output_dir, 'age_distribution.png')
    plt.savefig(age_dist_path)
    plt.close()
    print(f"Saved age distribution plot to {age_dist_path}")
    print("\n--- End of EDA ---")


if __name__ == '__main__':
    DATA_PATH = '../../dermatology/dermatology.data'
    REPORTS_DIR = '../../reports'
    
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    explore_data(DATA_PATH, REPORTS_DIR)
