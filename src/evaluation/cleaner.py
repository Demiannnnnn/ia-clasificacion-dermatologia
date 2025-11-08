import pandas as pd

def clean_data(df):
    """
    Cleans the dermatology dataframe by handling missing values.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    # Convert age column to numeric, coercing errors to NaN
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Fill missing age values with the mean of the column
    # This syntax avoids the FutureWarning
    df['age'] = df['age'].fillna(df['age'].mean())
    
    # Ensure all feature columns are of a numeric type for the model
    # The 'class' column is the target and can remain as is.
    for col in df.columns:
        if col != 'class':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional: Drop rows if any other column has NaNs after coercion
    df.dropna(inplace=True)

    return df

if __name__ == '__main__':
    # Example of how to use the function
    from src.data.loader import load_data
    data_path = '../../dermatology/dermatology.data'
    
    # In the new flow, loader just loads
    dermatology_df_raw = load_data(data_path)
    print("Info before cleaning:")
    dermatology_df_raw.info()
    
    # Cleaner handles the missing values
    dermatology_df_cleaned = clean_data(dermatology_df_raw.copy()) # Use copy to show difference
    
    print("\nInfo after cleaning:")
    dermatology_df_cleaned.info()
    print("\nNull values in 'age' column after cleaning:", dermatology_df_cleaned['age'].isnull().sum())
