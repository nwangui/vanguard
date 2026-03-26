#------- PYTHON BASED AI INTRUSION DETECTION SYSTEM USING THE CIC-IDS2017 DATASET --------#

import pandas as pd
import numpy as np
import zipfile
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# Prevent The Code From Crashing
# Silence the TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disables specialized CPU optimizations that cause locks

tf.config.set_visible_devices([], 'GPU') # Force CPU-only mode

# 3. Limit the number of threads to prevent the "Lock blocking"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

#------ LOADING THE CIC-IDS2017 DATASET -----#
def load_and_merge_zip(zip_path):
    all_df = []
    with zipfile.ZipFile('MachineLearningCSV.zip', 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.csv') and not file_name.startswith('__'):
                print(f"Reading: {file_name}")
                with z.open(file_name) as f:
                    all_df.append(pd.read_csv(f))
    return pd.concat(all_df, ignore_index=True)


df = load_and_merge_zip('MachineLearningCSV.zip')

#----- DATA CLEANING & INTEGRATION -----#
# Remove Trailing Spaces from Column Names
df.columns = df.columns.str.strip()

# Missing Values & Infinity
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Label Encoding - To convert text to numbers
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

#----- STRATIFIED SAMPLING -----#
# Using 20% so as not to overwhelm the machine learning model
X = df.drop('Label', axis=1)
y = df['Label']

X_sample, _, y_sample, _ = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42
)

# DATA SPLIT - TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

# STANDARD SCALER
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-Hot Encoding for Keras
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print("Pre-processing Complete.")
print(f"Final Training Shape: {X_train_scaled.shape}")


#------- DEEP NEURAL NETWORK (MLP) TRAINING --------#

# Define paths for your assets
model_path = 'models/ids_keras_model.h5'
scaler_path = 'models/scaler.pkl'
encoder_path = 'models/label_encoder.pkl'

# Initialize model as None (In order to avoid confusing the system)
model = None

#Check if we already have a trained model to avoid repeating work
if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
    print("\n✅ Existing model assets found. Loading...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)
    num_classes = len(le.classes_)

else:
    print("\n🚀 No existing model found. Starting training...")
    num_classes = len(le.classes_)

input_dim = X_train.shape[1] # Automatically detects your 78 features

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(64, activation='relu'),        # Hidden Layer 1
    Dropout(0.2),                        # 20% of neurons shut off randomly to prevent overfitting
    Dense(32, activation='relu'),        # Hidden Layer 2
    Dense(num_classes, activation='softmax') # Softmax gives a probability for each of the 15 classes
])

# Compile and Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Starting Keras Training ---")
model.fit(X_train_scaled, y_train_cat, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the Model
print("\n--- Model Evaluation ---")


# Model Prediction
print("\n--- Generating Predictions ---")
y_probs = model.predict(X_test_scaled)

# Convert Probabilities to Class Indices - To find the highest probability
y_pred = np.argmax(y_probs, axis=1)

# Display labels only in the train set
present_classes = np.unique(y_test)
target_names = le.inverse_transform(present_classes)

# Classification Report
print("\n--- IDS CLASSIFICATION REPORT (KERAS) ---")
print(classification_report(
    y_test,
    y_pred,
    labels=present_classes,
    target_names=target_names
))


# Create a directory for the deep neural network model
if not os.path.exists('models'):
    os.makedirs('models')

# Save the Keras model (Architecture + Weights + Optimizer state) - The .h5 format which is the standard for Keras
model.save('models/ids_keras_model.h5')

# Save the Scaler and LabelEncoder (Using Joblib)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("--- All Assets Saved Successfully ---")
print("Files created: ids_keras_model.h5, scaler.pkl, label_encoder.pkl")

#----- CONFUSION MATRIX -----#

def plot_keras_confusion_matrix(trained_model, x_val, y_val, label_enc):
    """
    Plots a heatmap for the IDS performance.
    'trained_model' is your Keras .h5 model.
    'x_val' is your X_test_scaled data.
    """
    # Generate probabilities
    predictions_probs = trained_model.predict(x_val)

    # Convert to class indices
    predictions_classes = np.argmax(predictions_probs, axis=1)

    # Create the matrix
    cm = confusion_matrix(y_val, predictions_classes)

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_enc.classes_,
                yticklabels=label_enc.classes_)

    plt.title('Deep Learning IDS: Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.show()

# Save the plot
    plt.savefig('ids_confusion_matrix.png')
    plt.show()

# Call the function
plot_keras_confusion_matrix(model, X_test_scaled, y_test, le)

