from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    """
    Scales the features using StandardScaler.

    Args:
        X_train (pandas.DataFrame): The training features.
        X_test (pandas.DataFrame): The testing features.

    Returns:
        tuple: A tuple containing the scaled X_train and X_test.
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

if __name__ == '__main__':
    # This is an example of how to use the function
    # We need to load and split the data first
    from src.data.loader import load_data
    from src.evaluation.splitter import split_data
    
    data_path = '../../dermatology/dermatology.data'
    dermatology_df = load_data(data_path)
    
    X_train, X_test, _, _ = split_data(dermatology_df)
    
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("X_train_scaled mean (first feature):", X_train_scaled[:, 0].mean())
    print("X_train_scaled std (first feature):", X_train_scaled[:, 0].std())
