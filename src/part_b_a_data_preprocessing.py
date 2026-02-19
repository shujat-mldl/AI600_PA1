import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_process_data(path='../train.csv'):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.read_csv(path.replace('../', ''))

    for col in ['room_type', 'neighbourhood_group']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    num_cols = ['amenity_score', 'availability_365', 'minimum_nights', 'number_of_reviews']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())


    df['minimum_nights'] = np.log1p(df['minimum_nights'])

    X = df.drop('price_class', axis=1)
    y = df['price_class'].values

    cat_cols = ['neighbourhood_group', 'room_type']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)
    
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
    
    return X_encoded.values, y

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    X_all, y_all = load_and_process_data('../train.csv')
    print("Data loaded successfully.")
    print(f"Shape: {X_all.shape}")