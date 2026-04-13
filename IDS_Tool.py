# ------- VANGUARD IDS: PYTORCH & RANDOM FOREST DUAL-ENGINE TRAINING (AUC-ROC BASED) --------#

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

print("🛡️ Vanguard IDS: Initializing Forensic Training Engine...")

# --- THREAT INTELLIGENCE MAPPING ---#
# Maps the abstract dataset labels to real-world forensic vulnerabilities (CVEs)
CVE_INTEL_MAP = {
    'DoS': 'CVE-2023-44487 (HTTP/2 Rapid Reset)',
    'PortScan': 'Reconnaissance / Network Enumeration',
    'Brute Force': 'CVE-2020-3580 (Cisco ASA Auth Bypass)',
    'Infiltration': 'CVE-2021-44228 (Log4Shell RCE)',
    'Web Attack': 'CVE-2022-22965 (Spring4Shell)',
    'Bot': 'CVE-2016-10372 (Mirai IoT Botnet (RCE))',
    'FTP-Patator': 'Protocol Brute Force (FTP)',
    'SSH-Patator': 'Protocol Brute Force (SSH)'
}

# --- LOADING DATASET & PRE-PROCESSING --- #
def load_and_merge_zip(zip_path):
    all_df = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__')]
        if not csv_files:
            raise ValueError("⚠️ The ZIP file contains no valid .csv network traffic logs.")
        for file_name in csv_files:
            print(f"Reading: {file_name}")
            with z.open(file_name) as f:
                all_df.append(pd.read_csv(f))
    return pd.concat(all_df, ignore_index=True)

df = load_and_merge_zip('MachineLearningCSV.zip')
df = df.loc[:, ~df.columns.duplicated()] #Remove duplicate columns if any files had overlapping headers
df.columns = df.columns.str.strip() #Removes leading/trailing spaces in column names
df.replace([np.inf, -np.inf], np.nan, inplace=True) #Replaces infinite values with NaN
df.dropna(inplace=True) #Removes rows with missing or infinite data to prevent model errors

#Converts text labels into numbers
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']

# Sampling 50% ensures the model trains quickly while maintaining attack distribution
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Standardizing features which is crucial for Neural Network convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors to allow the model to perform backpropagation
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for Batch Training
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

# --- PYTORCH NEURAL NETWORK --- #
print(f"\n🏆 Training PyTorch Baseline...")
class IDSNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.3) #Randomly shuts off 30% of neurons during training to prevent overfitting
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.relu = nn.ReLU() #Activation function to help the model learn non-linear patterns

    # This is the path data takes through the network:
    # input → layer 1 → activation → dropout → layer 2 → activation → output scores
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)


input_dim = scaler.n_features_in_
num_classes = len(le.classes_)
pytorch_model = IDSNetwork(input_dim, num_classes)

# PyTorch Training Loop
#CrossEntropyLoss is standard for multi-class classification
criterion = nn.CrossEntropyLoss()
#Adam optimizer adaptively adjusts learning rates
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
epochs = 5
pytorch_model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = pytorch_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

# PyTorch AUC-ROC Evaluation
pytorch_model.eval()
with torch.no_grad():
    logits = pytorch_model(X_test_tensor)
    pytorch_probs = torch.softmax(logits, dim=1).numpy()
    pytorch_preds = np.argmax(pytorch_probs, axis=1)

# Converts labels to a binary format (0/1) for each class to calculate ROC
y_test_binarized = label_binarize(y_test, classes=range(num_classes))

# AUC-ROC score: Measures the Area Under the Curve (AUC)
# Macro average to ensure that the evaluation remains unbiased
pytorch_auc = roc_auc_score(y_test_binarized, pytorch_probs, multi_class='ovr', average='macro')
pytorch_acc = accuracy_score(y_test, pytorch_preds)

print(f"\n🏆 PyTorch Metrics; Accuracy: {pytorch_acc:.4f} | AUC-ROC : {pytorch_auc:.4f}")
print("\n📋 PyTorch Classification Report:")
print(classification_report(y_test, pytorch_preds, target_names=le.classes_, zero_division=0))


# --- RANDOM FOREST CLASSIFIER --- #
print("\n🌲 Training Random Forest Baseline...")

#class_weight='balanced': The model pays more attention to rare attacks
rf_model = RandomForestClassifier(
    n_estimators=100, #More trees for a more stable consensus
    max_depth=15,     #Deep enough to see complex attack patterns
    min_samples_split=5, #Prevents trees from becoming too specific resulting in anti-overfitting
    random_state=42,
    class_weight='balanced' #Critical for an imbalanced dataset
)

rf_model.fit(X_train_scaled, y_train)

# Random Forest ROC-AUC Evaluation
rf_probs = rf_model.predict_proba(X_test_scaled)
rf_preds = rf_model.predict(X_test_scaled)
rf_auc = roc_auc_score(y_test_binarized, rf_probs, multi_class='ovr', average='macro')
rf_acc = accuracy_score(y_test, rf_preds)

print(f"🌲 RF Metrics -> Accuracy: {rf_acc:.4f} | AUC-ROC : {rf_auc:.4f}")
print("\n📋 Random Forest Classification Report:")
print(classification_report(y_test, rf_preds, target_names=le.classes_, zero_division=0))


# --- ENGINE AUC-ROC VISUALIZATION --- #
# Identify the index for 'BENIGN' and remove it from plotting to show actual attacks
benign_idx = list(le.classes_).index('BENIGN')

# Function to Plot specific Attack Categories
def plot_engine_roc(y_test_bin, y_probs, le_classes, engine_name, file_name):
    plt.figure(figsize=(10, 6))
    for i in range(len(le_classes)):
        if i == benign_idx: continue #Exclude Benign from attack analysis to focus specifically on attack detection
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i]) #Calculates False Positive Rate and True Positive Rate
        roc_auc_val = auc(fpr, tpr) #Draws the curve for each attack category
        plt.plot(fpr, tpr, label=f'Attack: {le_classes[i]} (AUC = {roc_auc_val:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
    plt.title(f'Vanguard IDS: {engine_name} ROC Analysis (Malicious Threats Only)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(alpha=0.3)
    plt.savefig(file_name)
    plt.show()

# PyTorch ROC Analysis
print("\nGenerating PyTorch ROC Analysis Chart...")
plot_engine_roc(y_test_binarized, pytorch_probs, le.classes_, "PyTorch Neural Net", 'pytorch_roc_attacks.png')

# Random Forest ROC Analysis
print("Generating Random Forest ROC Analysis Chart...")
plot_engine_roc(y_test_binarized, rf_probs, le.classes_, "Random Forest (Ensemble)", 'rf_roc_attacks.png')

# --- MACHINE LEARNING ENGINE SELECTION --- #
if not os.path.exists('models'): os.makedirs('models')

# AUC-ROC Comparison
print(f"\n📊 FINAL FORENSIC COMPARISON:")
print(f"🏆️ PyTorch AUC-ROC: {pytorch_auc:.4f}")
print(f"🌲 Random Forest AUC-ROC: {rf_auc:.4f}")

# Engine Selection Logic
if rf_auc > pytorch_auc:
    print("\n✅ Selection: Random Forest had superior AUC-ROC. Deploying RF as active engine...")
    joblib.dump(rf_model, 'models/vanguard_model.pkl')
    # Write the file for Streamlit (vanguard.py)
    with open('models/active_model_type.txt', 'w') as f:
        f.write('sklearn')
else:
    print("\n✅ Selection: PyTorch had superior AUC-ROC. Deploying PyTorch as active engine...")
    torch.save(pytorch_model.state_dict(), 'models/vanguard_model.pth')
    # Write the file for Streamlit (vanguard.py)
    with open('models/active_model_type.txt', 'w') as f:
        f.write('pytorch')

# Save universal assets (Scaler and LabelEncoder)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("🚀 Forensic Engine Update Complete. Best model saved in /models/")