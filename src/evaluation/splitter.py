from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        df (pandas.DataFrame): The input data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """
    X = df.drop('class', axis=1)
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # This is an example of how to use the function
    # We need to load the data first
    from src.data.loader import load_data
    data_path = '../../dermatology/dermatology.data'
    dermatology_df = load_data(data_path)
    
    X_train, X_test, y_train, y_test = split_data(dermatology_df)
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
