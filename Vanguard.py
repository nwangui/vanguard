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

st.title("🛡️ Vanguard: AI-Intrusion Detection System")
st.markdown(f"""
**Status:** {"☁️ Cloud Mode (Optimized)" if IS_CLOUD else "💻 Local Mode (Full)"}  
This framework dynamically selects the most robust engine (PyTorch or Random Forest) to mitigate **overfitting** and detect network anomalies.
""")


# --- 3. UPDATED PYTORCH ARCHITECTURE ---
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


# --- 4. DYNAMIC ASSET LOADING ---
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

# --- 5. DATA INGESTION ---
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

            # Dashboard Metrics
            st.subheader("🚨 Threat Detection Dashboard")
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Incident Summary")
                st.write(df['Threat_Type'].value_counts())

            with col2:
                st.write("#### Malicious Traffic Distribution")
                attack_only = df[df['Threat_Type'] != 'BENIGN']['Threat_Type'].value_counts()
                if not attack_only.empty:
                    st.bar_chart(attack_counts)
                else:
                    st.success("No malicious patterns detected in the network traffic logs.")

            # Alerting Logic
            malicious_df = df[df['Threat_Type'] != 'BENIGN']
            if not malicious_df.empty:
                st.error(f"🔥 ALERT: {len(malicious_df)} High-Risk entries identified.")
                st.dataframe(malicious_df)
            else:
                st.balloons()
                st.success("✅ System Status: Secure. All traffic identified as Benign.")

else:
    st.info("Awaiting input: Upload a network traffic CSV file to begin analysis.")

# --- 6. DISSERTATION FOOTER ---
st.markdown("---")
st.caption(" Intrusion Detection System Framework | Developed by Nicole Wangui Mbau | Middlesex University Dubai")


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

st.title("🛡️ Vanguard: AI-Intrusion Detection System")
st.markdown(f"""
**Status:** {"☁️ Cloud Mode (Optimized)" if IS_CLOUD else "💻 Local Mode (Full)"}  
This intrusion detection tool analyzes network traffic for threats using **PyTorch Deep Learning** engine.
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
    scaler_path = 'models/scaler.pkl'
    encoder_path = 'models/label_encoder.pkl'
    active_model_info = 'models/active_model_type.txt'

    # 1. Load the Universal Assets (Scaler and Labels)
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    # 2. Check which model was saved as the "Winner"
    with open(active_model_info, 'r') as f:
        model_type = f.read().strip()

    # 3. Dynamic Loading based on the Model Type
    if model_type == 'sklearn':
        # Load the Random Forest (.pkl)
        model = joblib.load('models/vanguard_model.pkl')
        is_pytorch = False
    else:
        # Load the PyTorch Model (.pth)
        input_dim = 78
        num_classes = len(le.classes_)
        model = IDSNetwork(input_dim, num_classes)
        model.load_state_dict(torch.load('models/vanguard_model.pth', map_location=torch.device('cpu')))
        model.eval()
        is_pytorch = True

    return model, scaler, le, is_pytorch

# Initialize the assets
with st.spinner("Vanguard is selecting the optimal engine..."):
    model, scaler, le, is_pytorch = load_vanguard_assets()
    st.sidebar.success(f"✅ Active Engine: {('PyTorch' if is_pytorch else 'Random Forest')}")

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

    st.subheader("📊 Traffic Preview")
    st.dataframe(df.head(10))

    if st.button("🔍 Run Traffic Analysis for malicious traffic"):
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

            if is_pytorch:
                # Logic for PyTorch
                input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_classes = torch.argmax(outputs, dim=1).numpy()
            else:
                # Logic for Random Forest (Sklearn)
                # It uses the scaled_data (numpy array) directly
                pred_classes = model.predict(scaled_data)

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
                st.write("#### Attack Visual Distribution")

                # Get the value counts of all detections
                all_counts = df['Threat_Type'].value_counts()

                # Filter out 'BENIGN' so the chart displays malicious traffic only
                attack_counts = all_counts.drop(labels=['BENIGN'], errors='ignore')

                # Check if there are is any malicious attack
                if not attack_counts.empty:
                    st.bar_chart(attack_counts)
                else:
                    st.info("No malicious traffic to visualize.")
                st.bar_chart(df['Threat_Type'].value_counts())

            # Highlight only Threats
            threats_only = df[df['Threat_Type'] != 'BENIGN']
            if not threats_only.empty:
                st.error(f"🔥 Alert: {len(threats_only)} malicious traffic detected!")
                st.dataframe(threats_only)
            else:
                st.success("✅ No threats detected in this traffic log.")

else:
    st.info("Please upload a network traffic CSV file in the sidebar to begin.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Developed by Nicole Wangui Mbau | MSc Cybersecurity & Emerging Threats | Middlesex University Dubai")