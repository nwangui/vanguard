#----- USER INTERFACE ON STREAMLIT THROUGH GITHUB -----#

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn

# --- 1. CLOUD OPTIMIZATION CHECK ---
IS_CLOUD = os.environ.get('RENDER') == 'true'
MAX_ROWS = 5000 if IS_CLOUD else None

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Vanguard IDS", page_icon="🛡️", layout="wide")

st.title("🛡️ Vanguard: AI-Driven Intrusion Detection")
st.markdown(f"""
**Status:** {"☁️ Cloud Mode (Optimized)" if IS_CLOUD else "💻 Local Mode (Full)"}  
This system analyzes network traffic for emerging threats using a **PyTorch Deep Learning** engine.
""")


# --- 3. PYTORCH MODEL ARCHITECTURE ---
# This MUST match the class you used in your IDS_Tool.py script
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
        return self.output(x)


# --- 4. LOAD ASSETS (WITH CACHING) ---
@st.cache_resource
def load_vanguard_assets():
    model_path = 'models/ids_pytorch_model.pth'
    scaler_path = 'models/scaler.pkl'
    encoder_path = 'models/label_encoder.pkl'

    # Load Scaler and Label Encoder
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    # Initialize PyTorch Model
    input_dim = 78  # CIC-IDS2017 standard features
    num_classes = len(le.classes_)
    model = IDSNetwork(input_dim, num_classes)

    # Load weights (map_location='cpu' ensures it works on Cloud/Mac regardless of GPU)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode

    return model, scaler, le


with st.spinner("Initializing AI Engine..."):
    try:
        model, scaler, le = load_vanguard_assets()
        st.sidebar.success("✅ PyTorch Engine Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()

# --- 5. USER INTERFACE ---
st.sidebar.header("Upload Traffic Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file (CIC-IDS2017 format)", type="csv")

if uploaded_file is not None:
    df_chunk = pd.read_csv(uploaded_file, nrows=100000)
    df_chunk.columns = df_chunk.columns.str.strip()

    if IS_CLOUD and len(df_chunk) > MAX_ROWS:
        df = df_chunk.sample(n=MAX_ROWS, random_state=42)
        st.warning(f"⚠️ Sampled {MAX_ROWS} rows for cloud performance.")
    else:
        df = df_chunk
        st.success(f"✅ Analyzing {len(df)} rows.")

    st.subheader("📊 Ingress Traffic Preview")
    st.dataframe(df.head(10))

    if st.button("🔍 Run Threat Analysis"):
        with st.spinner("Vanguard is scanning for anomalies..."):
            # Data Cleaning
            features = df.drop(columns=['Label'], errors='ignore')
            features = features.select_dtypes(include=[np.number])

            # Ensure correct number of features
            if features.shape[1] != 78:
                st.error(f"Feature mismatch! Expected 78, got {features.shape[1]}. Check CSV format.")
                st.stop()

            features.replace([np.inf, -np.inf], np.nan, inplace=True)
            features.fillna(0, inplace=True)

            # Preprocessing & Prediction
            scaled_data = scaler.transform(features)
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(input_tensor)
                pred_classes = torch.argmax(outputs, dim=1).numpy()

            # Mapping
            df['Prediction'] = pred_classes
            # Using your actual LabelEncoder to map numbers back to attack names
            df['Threat_Type'] = le.inverse_transform(pred_classes)

            # Results Display
            st.subheader("🚨 Analysis Results")
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Detection Summary")
                st.write(df['Threat_Type'].value_counts())

            with col2:
                st.write("#### Visual Distribution")
                st.bar_chart(df['Threat_Type'].value_counts())

            # Highlight only Threats
            threats_only = df[df['Threat_Type'] != 'BENIGN']
            if not threats_only.empty:
                st.error(f"🔥 Alert: {len(threats_only)} malicious activities detected!")
                st.dataframe(threats_only)
            else:
                st.success("✅ No threats detected in this traffic sample.")

else:
    st.info("Please upload a network traffic CSV file in the sidebar to begin.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Developed by Nicole Wangui Mbau | MSc Cybersecurity & Emerging Threats | Middlesex University Dubai")