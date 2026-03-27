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
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
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
        self.fc1 = nn.Linear(input_dim, 64) # The input layer takes the 78 features and passes them to 64 neurons.
        self.dropout = nn.Dropout(0.5) # 0.5 to prevent overfitting
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes) # The output layer provides 15 final scores for each possible attack type in the dataset.
        self.relu = nn.ReLU() # Decides which information is important enough to pass forward.

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

# Get probabilities instead of just the class prediction
model.eval()
with torch.no_grad():
    logits = model(X_test_tensor)
    # Apply softmax to get probabilities for each class
    y_probs = torch.softmax(logits, dim=1).numpy()

# Binarize the output for Multi-class ROC - This converts labels like [0, 1, 2] into [[1,0,0], [0,1,0], [0,0,1]]
y_test_binarized = label_binarize(y_test, classes=range(num_classes))

# Calculate the Macro-Average AUC-ROC - 'macro' treats all classes equally, which is great for finding rare attacks
roc_auc_macro = roc_auc_score(y_test_binarized, y_probs, multi_class='ovr', average='macro')

print(f"\n🏆 Overall AUC-ROC Score: {roc_auc_macro:.4f}")

plt.figure(figsize=(12, 8))

# 1. Get all class counts
all_class_counts = pd.Series(y_test).value_counts()

# 2. Find the index for 'BENIGN' and remove it from our plotting list
benign_idx = list(le.classes_).index('BENIGN')
attack_counts = all_class_counts.drop(index=benign_idx, errors='ignore')

# 3. Pick the Top 5 remaining actual attack classes
top_5_attack_indices = attack_counts.head(5).index.tolist()

# 4. Plotting the ROC Curves for Attacks
for i in top_5_attack_indices:
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)

    class_name = le.classes_[i]
    plt.plot(fpr, tpr, lw=2.5, label=f'Attack: {class_name} (AUC = {roc_auc:.4f})')

# 5. Formatting for your Thesis
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Vanguard IDS: ROC Analysis of Top 5 Malicious Threats')
plt.legend(loc='lower right', fontsize='medium')
plt.grid(alpha=0.3)

# 6. Save high-res version
plt.savefig('vanguard_roc_attacks_only.png', dpi=300)
plt.show()

# ------ CONFUSION MATRIX -----#
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('PyTorch IDS: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('ids_confusion_matrix_pytorch.png')
plt.show()


#----- RANDOM FOREST AI INTRUSION DETECTION SYSTEM (CIC-IDS2017 University of New Brunswick) -----#

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load thr pre-processed data (Assuming X_train_scaled, y_train are ready)

print("🌲 Initializing Constrained Random Forest...")

# Limit max_depth to 3 to prevent overfitting or data leakage
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    random_state=42,
    class_weight='balanced'
)

# Train the Model
rf_model.fit(X_train_scaled, y_train)

# Evaluate the Model
rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)

print(f"📊 Random Forest Accuracy: {rf_acc:.4f}")
print("\n--- RF Classification Report ---")
print(classification_report(y_test, rf_preds))