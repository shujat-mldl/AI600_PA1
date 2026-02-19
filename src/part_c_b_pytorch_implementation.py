import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

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

    return X_encoded.values, y, X_encoded.columns.tolist()

if __name__ == "__main__":
    print("Loading data...")
    X_all, y_all, feature_names = load_and_process_data('../train.csv')
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    class PyTorchMLP(nn.Module):
        def __init__(self, input_dim):
            super(PyTorchMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 4)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x

    model = PyTorchMLP(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5. Train the Model
    print("Training PyTorch Model for Feature Attribution...")
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    print(f"Training Complete. Final Loss: {loss.item():.4f}")

    print("\nCalculating Gradient-Based Feature Attribution...")

    X_train_tensor.requires_grad = True

    model.zero_grad()
    outputs = model(X_train_tensor)
    final_loss = criterion(outputs, y_train_tensor)
    final_loss.backward()

    input_gradients = X_train_tensor.grad
    importance_scores = torch.mean(torch.abs(input_gradients), dim=0).detach().numpy()

    feature_ranking = pd.DataFrame({
        'Feature': feature_names, 
        'Importance': importance_scores
    })
    feature_ranking = feature_ranking.sort_values(by='Importance', ascending=False)

    print("\n--- Ranked List of Influential Features ---")
    print(feature_ranking.to_string(index=False))

    os.makedirs('../plots', exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.barh(feature_ranking['Feature'][::-1], feature_ranking['Importance'][::-1], color='purple')
    plt.xlabel('Average Absolute Gradient Magnitude $|\partial L / \partial x_i|$')
    plt.title('Feature Importance via Gradient Attribution (PyTorch)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../plots/feature_importance_pytorch.png', dpi=300)
    plt.show()

    print("\nPlot saved to ../plots/feature_importance_pytorch.png")