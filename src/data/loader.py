import pandas as pd
from numpy import nan

def load_data(file_path):
    """
    Loads the dermatology data from a CSV file.

    Args:
        file_path (str): The path to the dermatology.data file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    column_names = [
        'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
        'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement',
        'knee_and_elbow_involvement', 'scalp_involvement', 'family_history',
        'melanin_incontinence', 'eosinophils_in_the_infiltrate', 'pnl_infiltrate',
        'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis',
        'hyperkeratosis', 'parakeratosis', 'clubbing_of_the_rete_ridges',
        'elongation_of_the_rete_ridges', 'thinning_of_the_suprapapillary_epidermis',
        'spongiform_pustule', 'munro_microabcess', 'focal_hypergranulosis',
        'disappearance_of_the_granular_layer', 'vacuolisation_and_damage_of_basal_layer',
        'spongiosis', 'saw-tooth_appearance_of_retes', 'follicular_horn_plug',
        'perifollicular_parakeratosis', 'inflammatory_monoluclear_infiltrate',
        'band-like_infiltrate', 'age', 'class'
    ]
    
    df = pd.read_csv(file_path, header=None, names=column_names, na_values='?')
    
    # Convert age column to numeric, coercing errors to NaN
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Fill missing age values with the mean
    df['age'].fillna(df['age'].mean(), inplace=True)
    
    return df

if __name__ == '__main__':
    # Example of how to use the function
    data_path = '../dermatology/dermatology.data'
    dermatology_df = load_data(data_path)
    print(dermatology_df.head())
    print(dermatology_df.info())

