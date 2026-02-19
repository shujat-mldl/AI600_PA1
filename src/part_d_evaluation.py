import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

modes = train_df[['room_type', 'neighbourhood_group']].mode().iloc[0]
medians = train_df[['amenity_score', 'availability_365', 'minimum_nights', 'number_of_reviews']].median()

for df in [train_df, test_df]:
    for col in ['room_type', 'neighbourhood_group']:
        df[col] = df[col].fillna(modes[col])
    for col in medians.index:
        df[col] = df[col].fillna(medians[col])
    df['minimum_nights'] = np.log1p(df['minimum_nights'])

y_train = train_df['price_class'].values
X_train_raw = train_df.drop('price_class', axis=1)

y_test = test_df['price_class'].values
X_test_raw = test_df.drop('price_class', axis=1)

cat_cols = ['neighbourhood_group', 'room_type']
X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=False, dtype=float)
X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=False, dtype=float)

X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

num_cols = ['amenity_score', 'availability_365', 'minimum_nights', 'number_of_reviews']
scaler = StandardScaler()
X_train_scaled = X_train_enc.copy()
X_test_scaled = X_test_enc.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train_enc[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test_enc[num_cols]) # <--- CRITICAL: transform only!

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

model = PyTorchMLP(input_dim=X_train_scaled.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Training model...")
for epoch in range(200):
    optimizer.zero_grad()
    loss = criterion(model(X_train_tensor), y_train_tensor)
    loss.backward()
    optimizer.step()

model.eval() 
with torch.no_grad(): 
    train_preds = torch.argmax(model(X_train_tensor), dim=1).numpy()
    test_preds = torch.argmax(model(X_test_tensor), dim=1).numpy()

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"\n--- Final Accuracies ---")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print(f"Performance Gap:   {(train_acc - test_acc)*100:.2f}%")