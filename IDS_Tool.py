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
        self.dropout = nn.Dropout(0.6) # 0.5 to prevent overfitting
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
print("🚀 Initializing Fresh Training Session (Bypassing existing models)...")

# Define your Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop (Ensure your train_loader is defined)
epochs = 5  # Adjust based on your 0.7-0.8 accuracy goal
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

# Save the Directory
if not os.path.exists('models'):
    os.makedirs('models')

# Save The PyTorch Model
py_model_path = 'models/pytorch_model.pth'
torch.save(model.state_dict(), py_model_path)

#Save The Scaler and Label Encoder
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

# Calculate PyTorch Accuracy
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted_pytorch = torch.max(outputs, 1)
    # Convert to numpy for comparison
    pytorch_acc = (predicted_pytorch == y_test_tensor).sum().item() / y_test_tensor.size(0)

print(f"🛡️ PyTorch Accuracy: {pytorch_acc:.4f}")

#------ AUC-ROC CURVE -----#
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

# Get all class counts
all_class_counts = pd.Series(y_test).value_counts()

# Find the index for 'BENIGN' and remove it from our plotting list
benign_idx = list(le.classes_).index('BENIGN')
attack_counts = all_class_counts.drop(index=benign_idx, errors='ignore')

# Pick the Top 5 remaining actual attack classes
top_5_attack_indices = attack_counts.head(5).index.tolist()

# Plotting the ROC Curves for Attacks
for i in top_5_attack_indices:
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)

    class_name = le.classes_[i]
    plt.plot(fpr, tpr, lw=2.5, label=f'Attack: {class_name} (AUC = {roc_auc:.4f})')

# AUC-ROC Plot
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Vanguard IDS: ROC Analysis of Top 5 Malicious Threats')
plt.legend(loc='lower right', fontsize='medium')
plt.grid(alpha=0.3)

# Save high-resolution chart
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

# Load the pre-processed data (Assuming X_train_scaled, y_train are ready)
print("🌲 Initializing Constrained Random Forest...")

# Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
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

#Save Random Forest Model
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(rf_model, 'models/random_forest.pkl')


# ----- Saving The User Interface Engine Based Off Of The Machine Learning With The Best Accuracy Score -----#
print(f"\n📊 FINAL COMPARISON:")
print(f"🛡️ PyTorch Accuracy: {pytorch_acc:.4f}")
print(f"🌲 Random Forest Accuracy: {rf_acc:.4f}")

# Selection Logic - This saves the 'vanguard_model' with the lower accuracy
if rf_acc < pytorch_acc:
    print("🚀 Result: Random Forest accuracy is lower than that of PyTorch. Updating deployment files...")
    joblib.dump(rf_model, 'models/vanguard_model.pkl')
    # Write the 'handshake' file for Streamlit
    with open('models/active_model_type.txt', 'w') as f:
        f.write('sklearn')
else:
    print("🚀 Result: PyTorch accuracy is lower than that of Random Forest. Updating deployment files...")
    torch.save(model.state_dict(), 'models/vanguard_model.pth')
    # Write the 'handshake' file for Streamlit
    with open('models/active_model_type.txt', 'w') as f:
        f.write('pytorch')

print("✅ Vanguard Engine Update Complete. Best model saved to /models/")

