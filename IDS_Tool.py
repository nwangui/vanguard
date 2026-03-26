# ------- PYTORCH BASED AI INTRUSION DETECTION SYSTEM (CIC-IDS2017 University of New Brunswick) --------#

import os
import pandas as pd
import numpy as np
import zipfile
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("🛡️ Vanguard IDS: Initializing PyTorch Engine...")

# ------ 1. LOADING & PRE-PROCESSING -----#
def load_and_merge_zip(zip_path):
    all_df = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv') and not file_name.startswith('__'):
                print(f"Reading: {file_name}")
                with z.open(file_name) as f:
                    all_df.append(pd.read_csv(f))
    return pd.concat(all_df, ignore_index=True)


df = load_and_merge_zip('MachineLearningCSV.zip')
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

# Sampling 50% for performance stability
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for Batch Training
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)


# ------ NEURAL NETWORK ARCHITECTURE (PYTORCH) -----#
class IDSNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)  # CrossEntropyLoss handles the Softmax internally


input_dim = X_train_scaled.shape[1]
num_classes = len(le.classes_)
model = IDSNetwork(input_dim, num_classes)

# ------ TRAINING THE DATASET -----#
model_path = 'models/ids_pytorch_model.pth'

if os.path.exists(model_path):
    print("✅ Loading existing PyTorch model...")
    model.load_state_dict(torch.load(model_path))
else:
    print("🚀 No model found. Starting training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    # Save the Scaler and LabelEncoder (Using Joblib)
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')

# ------ EVALUATION & PREDICTION -----#
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    y_pred = torch.argmax(test_outputs, dim=1).numpy()

print("\n--- IDS CLASSIFICATION REPORT (PYTORCH) ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0  # This tells it to just show '0' instead of a warning
))

# ------ CONFUSION MATRIX -----#
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('PyTorch IDS: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('ids_confusion_matrix_pytorch.png')
plt.show()

# ------ ASSET EXPORT FOR GITHUB ------ #
print("\n📦 Exporting final assets for deployment...")

if not os.path.exists('models'):
    os.makedirs('models')

# Save the Scaler (Pre-processing brain)
joblib.dump(scaler, 'models/scaler.pkl')

# Save the Label Encoder (Classification brain)
joblib.dump(le, 'models/label_encoder.pkl')

# Save the PyTorch Model (Neural brain)
torch.save(model.state_dict(), 'models/ids_pytorch_model.pth')

print("✅ Assets verified: models/scaler.pkl, models/label_encoder.pkl, models/ids_pytorch_model.pth")