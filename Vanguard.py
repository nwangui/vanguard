#----- USER INTERFACE ON STREAMLIT THROUGH GITHUB -----#

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn


# --- SESSION STATE INITIALIZATION ---#
# This must be near the top to ensure the "Vault" exists before the app runs
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None


# --- CLOUD OPTIMIZATION CHECK --- #
IS_CLOUD = os.environ.get('RENDER') == 'true'
MAX_ROWS = 5000 if IS_CLOUD else None

# --- PAGE CONFIGURATION --- #
st.set_page_config(page_title="Vanguard IDS", page_icon="🛡️", layout="wide")

st.title("🛡️ Vanguard: AI-Intrusion Detection System")
st.markdown(f"""
**Status:** {"☁️ Cloud Mode (Optimized)" if IS_CLOUD else "💻 Local Mode (Full)"}  
This framework dynamically selects the most robust engine (PyTorch or Random Forest) to mitigate **overfitting** and detect network anomalies.
""")


# --- PYTORCH ARCHITECTURE ---#
# Matches the IDS_Tool.py changes (Dropout 0.6)
class IDSNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IDSNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.6)  # Increased to 0.6 for better generalization
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)


# --- ENGINE ASSET LOADING --- #
@st.cache_resource
def load_vanguard_assets():
    scaler_path = 'models/scaler.pkl'
    encoder_path = 'models/label_encoder.pkl'
    active_model_info = 'models/active_model_type.txt'

    # Load Universal Preprocessing Assets
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    # Read the Strategic Selection Handshake
    if os.path.exists(active_model_info):
        with open(active_model_info, 'r') as f:
            model_type = f.read().strip()
    else:
        model_type = 'pytorch'  # Default fallback

    # Load the "Underdog" Winner
    if model_type == 'sklearn':
        model = joblib.load('models/vanguard_model.pkl')
        is_pytorch = False
    else:
        input_dim = 78
        num_classes = len(le.classes_)
        model = IDSNetwork(input_dim, num_classes)
        # Load weights into the architecture
        model.load_state_dict(torch.load('models/vanguard_model.pth', map_location=torch.device('cpu')))
        model.eval()
        is_pytorch = True

    return model, scaler, le, is_pytorch

# Initialize the assets
with st.spinner("Vanguard is initializing the Detection Engine..."):
    try:
        model, scaler, le, is_pytorch = load_vanguard_assets()
        engine_name = "PyTorch Deep Learning" if is_pytorch else "Random Forest (Ensemble)"
        st.sidebar.success(f"✅ Active Engine: {engine_name}")
    except Exception as e:
        st.error(f"Error loading models: {e}. Ensure the /models folder is uploaded to GitHub.")
        st.stop()


# --- DATA INGESTION ---#
st.sidebar.header("Upload Traffic Data")
uploaded_file = st.sidebar.file_uploader("Upload Network CSV (CIC-IDS2017)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Optimization for Cloud Deployment
    if IS_CLOUD and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        st.warning(f"⚠️ Cloud Optimization: Analyzing {MAX_ROWS} sampled rows.")

    st.subheader("📊 Network Traffic Log Preview")
    st.dataframe(df.head(10))

    if st.button("🔍 Run Forensic Analysis"):
        with st.spinner("Analyzing traffic patterns..."):
            # Feature Extraction
            X_input = df.drop(columns=['Label'], errors='ignore')
            X_input = X_input.select_dtypes(include=[np.number])

            # Forensic Cleaning (Handles Inf/NaN common in PCAP-to-CSV)
            X_input.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_input.fillna(0, inplace=True)

            # Scale Data
            X_scaled = scaler.transform(X_input)

            # Perform Inference
            if is_pytorch:
                input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(input_tensor)
                    predictions = torch.argmax(logits, dim=1).numpy()
            else:
                predictions = model.predict(X_scaled)

            # Map Results
            df['Threat_Type'] = le.inverse_transform(predictions)

            # Session State Save
            st.session_state.analysis_results = df

# Dashboard Metrics
if st.session_state.analysis_results is not None:
    res_df = st.session_state.analysis_results

    st.markdown("---")
    st.subheader("🚨 Threat Detection Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Incident Summary")
        st.write(res_df['Threat_Type'].value_counts())

    with col2:
        st.write("#### Malicious Traffic Distribution")

        # Filter for attacks and get their counts
        attack_counts = res_df[res_df['Threat_Type'] != 'BENIGN']['Threat_Type'].value_counts()

        # Check if the series is not empty
        if not attack_counts.empty:
                st.bar_chart(attack_counts)
        else:
            st.success("No malicious patterns detected in the network traffic logs.")

        # Alerting Logic
        malicious_df = res_df[res_df['Threat_Type'] != 'BENIGN']
        if not malicious_df.empty:
            st.error(f"🔥 ALERT: {len(malicious_df)} High-Risk entries identified.")
            st.dataframe(malicious_df)
        else:
            st.balloons()
            st.success("✅ System Status: Secure. All traffic identified as Benign.")

        # Add a clear button for your demo
        if st.button("🗑️ Clear Results"):
            st.session_state.analysis_results = None
            st.rerun()
else:
    st.info("Awaiting input: Upload a network traffic CSV file to begin analysis.")

# --- DISSERTATION FOOTER ---#
st.markdown("---")
st.caption(" Intrusion Detection System Framework | Developed by Nicole Wangui Mbau | MSc Cybersecurity & Emerging Threats | Middlesex University Dubai")