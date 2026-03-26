import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # This hides all but the most critical errors

# --- 1. CLOUD OPTIMIZATION CHECK ---
# Render sets 'RENDER' to 'true' automatically if you added it as an Env Var
IS_CLOUD = os.environ.get('RENDER') == 'true'
MAX_ROWS = 5000 if IS_CLOUD else None

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Vanguard IDS", page_icon="🛡️", layout="wide")

st.title("🛡️ Vanguard: AI-Driven Intrusion Detection")
st.markdown(f"""
**Status:** {"☁️ Cloud Mode (Optimized)" if IS_CLOUD else "💻 Local Mode (Full)"}  
This system analyzes network traffic for emerging threats using a Deep Learning Keras model.
""")


# --- 3. LOAD ASSETS (WITH CACHING) ---
@st.cache_resource
def load_vanguard_assets():
    # Paths relative to your GitHub root
    model_path = 'models/ids_keras_model.h5'
    scaler_path = 'models/scaler.pkl'

    # Load model (using CPU-specific settings for Render)
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


with st.spinner("Initializing AI Engine..."):
    try:
        model, scaler = load_vanguard_assets()
        st.sidebar.success("✅ Model Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")

# --- 4. USER INTERFACE ---
st.sidebar.header("Upload Traffic Data")
uploaded_file = st.sidebar.file_saver = st.file_uploader("Choose a CSV file (CIC-IDS2017 format)", type="csv")

if uploaded_file is not None:
    # 1. Read the file (we use a chunk to avoid loading the whole 50GB at once)
    # We read a large enough chunk to sample from
    df_chunk = pd.read_csv(uploaded_file, nrows=100000)
    # Strip spaces from column names to ensure feature matching
    df_chunk.columns = df_chunk.columns.str.strip()

    # 2. Apply Random Sampling if in Cloud Mode
    if IS_CLOUD and len(df_chunk) > MAX_ROWS:
        df = df_chunk.sample(n=MAX_ROWS, random_state=42)
        st.warning(f"⚠️ Large file detected. Vanguard is analyzing a random sample of {MAX_ROWS} rows for performance.")
    else:
        df = df_chunk
        st.success(f"✅ Analyzing all {len(df)} rows.")

    st.subheader("📊 Ingress Traffic Preview (Sampled)")
    st.dataframe(df.head(10))

    if st.button("🔍 Run Threat Analysis"):
        with st.spinner("Vanguard is scanning for anomalies..."):
            # 1. Preprocessing - We drop 'Label' if it exists because the AI shouldn't see the "answer" yet
            features = df.drop(columns=['Label'], errors='ignore')
            features = features.select_dtypes(include=[np.number])

            # Ensure features match the 78 columns your model expects
            if features.shape[1] != model.input_shape[1]:
                st.error(
                    f"Feature mismatch! Expected {model.input_shape[1]} features, but got {features.shape[1]}. Check your CSV format.")
                st.stop()

            # Replace any 'inf' or '-inf' with NaN
            features.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Fill those NaNs (and any other missing values) with 0
            # (Or use features.median() if you want to be more precise)
            features.fillna(0, inplace=True)

            # Clip extreme values to prevent 'float64' overflow
            # (This caps values at a very high number instead of infinity)
            features = features.clip(lower=-1e15, upper=1e15)

            # --- NOW RUN THE SCALER ---
            scaled_data = scaler.transform(features)
            # 2. Scaling
            scaled_data = scaler.transform(features)

            # 3. Prediction
            predictions = model.predict(scaled_data)
            pred_classes = np.argmax(predictions, axis=1)

            # 4. Display Results
            st.subheader("🚨 Analysis Results")
            df['Prediction'] = pred_classes

            # Mapping numeric predictions back to attack names (Example mapping)
            # Update this list based on your specific training labels!
            attack_map = {0: "Benign", 1: "DDoS", 2: "PortScan", 3: "Bot", 4: "Web Attack"}
            df['Threat_Type'] = df['Prediction'].map(attack_map)

            col1, col2 = st.columns(2)
            with col1:
                st.write("#### Detection Summary")
                st.write(df['Threat_Type'].value_counts())

            with col2:
                st.write("#### Visual Distribution")
                st.bar_chart(df['Threat_Type'].value_counts())

            st.success("Analysis Complete.")
else:
    st.info("Please upload a network traffic CSV file in the sidebar to begin.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Developed by Nicole Wangui Mbau | MSc Cybersecurity & Emerging Threats | Middlesex University Dubai")