#----- USER INTERFACE ON STREAMLIT THROUGH GITHUB -----#

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# --- CVE MAPPING --- #
CVE_INTEL_BASE = {
    'DoS': {
        'cve': 'CVE-2023-44487',
        'name': 'HTTP/2 Rapid Reset',
        'severity': 7.5,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H',
        'description': 'Exploits a flaw in the HTTP/2 protocol stream cancellation to cause resource exhaustion.'
    },
    'PortScan': {
        'cve': 'CVE-2021-41773',
        'name': 'Network Enumeration (CWE-200)',
        'severity': 3.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N',
        'description': 'Identifying active services. This is the precursor to an exploit like the Apache Path Traversal.'
    },
    'Brute Force': {
        'cve': 'CVE-2020-3580',
        'name': 'Cisco ASA Auth Bypass',
        'severity': 6.1,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N',
        'description': 'Attempting unauthorized access via credential stuffing on administrative interfaces.'
    },
    'Infiltration': {
        'cve': 'CVE-2021-44228',
        'name': 'Log4Shell (RCE)',
        'severity': 10.0,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H',
        'description': 'Critical Remote Code Execution (RCE) via Log4j. Highly pervasive infiltration risk.'
    },
    'Web Attack': {
        'cve': 'CVE-2022-22965',
        'name': 'Spring4Shell',
        'severity': 9.8,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
        'description': 'RCE in Spring Framework via data binding, targeting web-facing applications.'
    },
    'Botnet': {
        'cve': 'CVE-2016-10372',
        'name': 'Mirai Variant',
        'severity': 7.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:H',
        'description': 'Infecting IoT devices to facilitate large-scale DDoS attacks.'
    },
    'FTP-Patator': {
        'cve': 'CVE-2025-49195',
        'name': 'FTP Brute Force (CWE-307)',
        'severity': 5.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N',
        'description': 'Automated authentication attacks against the File Transfer Protocol. Exploits CWE-307 (Improper Restriction of Excessive Authentication Attempts) due to a lack of login rate-limiting.'
    },
    'SSH-Patator': {
        'cve': 'CVE-2020-1616',
        'name': 'SSH Brute Force (CWE-307)',
        'severity': 5.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N',
        'description': 'Automated authentication attacks against the Secure Shell protocol. Exploits CWE-307 (Improper Restriction of Excessive Authentication Attempts) to gain unauthorized shell access.'
    }
}

# --- SESSION STATE INITIALIZATION ---#
# This must be near the top to ensure the "Vault" exists before the app runs
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None


# --- CLOUD OPTIMIZATION CHECK --- #
IS_CLOUD = os.environ.get('RENDER') == 'true'
MAX_ROWS = 5000 if IS_CLOUD else None

# --- PAGE CONFIGURATION --- #
st.set_page_config(page_title="Vanguard IDS", page_icon="🛡️", layout="wide")

st.markdown("""
    <div style="background-color:#0e1117; padding:20px; border-radius:10px; border: 1px solid #30363d; border-left: 5px solid #ffffff;">
        <h2 style="color:#ffffff; margin-top:0; font-family: sans-serif;">🛡️ Vanguard: AI-Intrusion Detection System</h2>
        <p style="color:#ffffff; font-size:15px; line-height:1.6;">
            A forensic intelligence tool that utilizes dual-engine machine learning to bridge the gap between automated detection and human triage 
            by mapping real-time behavioral anomalies to standardized CVSS 3.1 intelligence and MITRE CVE references.
        </p>
        <div style="display: flex; gap: 15px; font-family: monospace; font-size: 11px; margin-top: 10px;">
            <span style="color:#3fb950; background-color:rgba(63,185,80,0.1); padding:2px 8px; border-radius:5px;">● SYSTEM ONLINE</span>
            <span style="color:#58a6ff; background-color:rgba(88,166,255,0.1); padding:2px 8px; border-radius:5px;">● CVSS v3.1 MAPPING</span>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)


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
    st.subheader("🚨 Threat Analysis Dashboard")
    col1, col2 = st.columns([1,2])

    with col1:
        st.write("#### Incident Summary")
        st.write(res_df['Threat_Type'].value_counts())

    with col2:
        st.write("#### Malicious Traffic Distribution")

        # Filter for attacks
        malicious_df = res_df[res_df['Threat_Type'] != 'BENIGN']

        # Get counts and transform into a DataFrame - .reset_index() turns the Threat Names into a real column
        attack_counts = malicious_df['Threat_Type'].value_counts().reset_index()

        # Explicitly name the columns for the chart to reference
        attack_counts.columns = ['Threat Type', 'Count']

        # 4. Check if the dataframe is not empty
        if not attack_counts.empty:
            # We pass 'x' and 'y' using the new column names
            st.bar_chart(data=attack_counts, x='Threat Type', y='Count', color='#2162db')
        else:
            st.success("No malicious patterns detected in the network traffic logs.")

    st.markdown("---")
    col3, col4 = st.columns([1, 2])

    with col3:
        st.write("#### Risk Prioritization")
        # Logic to map unique threats to their CVSS scores
        unique_malicious = [t for t in res_df['Threat_Type'].unique() if t != 'BENIGN']

        severity_data = []
        for threat in unique_malicious:
            score = CVE_INTEL_BASE.get(threat, {}).get('severity', 0)
            severity_data.append({'Threat': threat, 'CVSS Score': score})

        if severity_data:
            sev_df = pd.DataFrame(severity_data).sort_values(by='CVSS Score', ascending=False)
            st.dataframe(sev_df.set_index('Threat'))
        else:
            st.write("No critical threats.")

    with col4:
        st.write("#### Severity Ranking (CVSS 3.1)")
        if severity_data:
            # Displaying the severity as a horizontal bar chart for clarity
            st.bar_chart(data=sev_df, x='Threat', y='CVSS Score', color='#ff4b4b')
        else:
            st.info("Awaiting threat data for risk visualization.")

    # --- Executive Summary ---#
    st.markdown("---")
    # --- Non-Technical Risk Translator---#
    RISK_TRANSLATOR = {
        'DoS': "an attempt to overwhelm our services and knock the system offline",
        'Infiltration': "a digital break-in where an attacker is trying to gain internal access to sensitive files",
        'Brute Force': "a coordinated attempt to guess employee passwords and take over accounts",
        'PortScan': "an attacker 'casing' our network to find an unlocked digital door",
        'Web Attack': "an attempt to exploit a weakness in our public-facing website",
        'Bot': "an infected network of devices attempting to use our resources for malicious activity",
        'FTP-Patator': "a repetitive attempt to break into our file storage systems",
        'SSH-Patator': "a repetitive attempt to gain remote control over our servers"
    }

    if st.button("📝 Generate Executive Summary (Non-Technical)"):
        malicious_df = res_df[res_df['Threat_Type'] != 'BENIGN']
        total_alerts = len(malicious_df)

        if total_alerts > 0:
            # A temporary list of detected threats and their scores
            detected_threats = malicious_df['Threat_Type'].unique()

            # Find the threat with the highest CVSS score
            top_threat = max(detected_threats, key=lambda x: CVE_INTEL_BASE.get(x, {}).get('severity', 0))
            max_severity = CVE_INTEL_BASE.get(top_threat, {}).get('severity', 0)

            st.warning(f"### 📋 Executive Summary: Action Required")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Highest Risk Level", f"{max_severity}/10",
                          delta="Critical" if max_severity > 8 else "High", delta_color="inverse")
            col_m2.metric("Total Anomalies", total_alerts)

            # Get translation for the highest risk threat
            risk_explanation = RISK_TRANSLATOR.get(top_threat, f"unusual activity")

            st.write(f"""
            **Current Status:** Vanguard has identified **{total_alerts}** unusual activities that deviate from the secure network baseline.

            **Primary Concern:** The most critical threat detected is **{top_threat}**. 

            **What this means:** This suggests **{risk_explanation}**. Even a single instance of this activity could lead to service outages, unauthorized access to private data, or a total compromise of our system's integrity.

            **Bottom Line:** An analyst should prioritize investigating the **{top_threat}** alerts immediately!
            """)
        else:
            st.success("### ✅ Executive Summary: System Secure")
            st.write("""
            **Current Status:** The analyzed traffic shows no unusual patterns. 

            **Conclusion:** All network behavior aligns with the secure baseline. No suspicious login attempts, break-ins, or service disruptions were detected. No further action is required from management at this time.
            """)

    # Alerting Logic
    st.markdown("---")
    st.write("### 💡Threat Intelligence (CVE Mapping)")

    malicious_df = res_df[res_df['Threat_Type'] != 'BENIGN']
    if not malicious_df.empty:
        unique_threats = malicious_df['Threat_Type'].unique()

        for threat in unique_threats:
            intel = CVE_INTEL_BASE.get(threat, None)
            if intel:
                with st.expander(f"🔍 Forensic Analysis: {threat} ({intel['cve']})"):
                    c_left, c_right = st.columns([3, 1])
                    with c_left:
                        st.write(f"**Vulnerability Name:** {intel['name']}")
                        st.write(f"**Vector:** {intel['vector']}")
                        st.write(f"**Description:** {intel['description']}")
                    with c_right:
                        st.metric(label="Risk Level", value=f"{intel['severity']}/10")

                    st.warning( f"**Action Plan:** {intel.get('action', 'Monitor traffic logs and restrict source IP.')}")

                    cve_id = intel.get('cve')
                    if cve_id:
                        st.markdown(f"[🔗 View Official MITRE Advisory for {cve_id}](https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve_id})")
    else:
        st.balloons()
        st.success("✅ System Status: Secure. All traffic identified as Benign.")

else:
    st.info("Awaiting input: Upload a network traffic CSV file to begin analysis.")

# --- DISSERTATION FOOTER ---#
st.markdown("---")
st.caption(" Intrusion Detection System Framework | Developed by Nicole Wangui Mbau | MSc Cybersecurity & Emerging Threats | Middlesex University Dubai")