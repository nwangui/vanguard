#----- USER INTERFACE ON STREAMLIT THROUGH GITHUB -----#

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn

# --- CVE MAPPING --- #
# A dictionary acting as a local threat database. Each key is an attack label and its value is another dictionary containing the CVE ID, severity score, CVSS vector, description, and action plan.
CVE_INTEL_BASE = {
    'DoS': {
        'cve': 'CVE-2023-44487',
        'name': 'HTTP/2 Rapid Reset',
        'severity': 7.5,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H',
        'description': 'Exploits a flaw in the HTTP/2 protocol stream cancellation to cause resource exhaustion.',
        'action': 'Rate-limit or block the source IP immediately and update server software to patched versions.',
    },
    'PortScan': {
        'cve': 'CVE-2021-41773',
        'name': 'Network Enumeration (CWE-200)',
        'severity': 3.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N',
        'description': 'Identifying active services. This is the precursor to an exploit like the Apache Path Traversal.',
        'action': 'Audit and close all non-essential open ports. Enable port-scan detection on your IDS/IPS and treat this as a precursor.'
    },
    'Brute Force': {
        'cve': 'CVE-2020-3580',
        'name': 'Cisco ASA Auth Bypass',
        'severity': 6.1,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:L/I:L/A:N',
        'description': 'Attempting unauthorized access via credential stuffing on administrative interfaces.',
        'action': 'Temporarily block the offending IP. Enforce account lockout after 5 failed attempts. Mandate MFA on all administrative interfaces.'
    },
    'Infiltration': {
        'cve': 'CVE-2021-44228',
        'name': 'Log4Shell (RCE)',
        'severity': 10.0,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H',
        'description': 'Critical Remote Code Execution (RCE) via Log4j. Highly pervasive infiltration risk.',
        'action': 'Isolate the affected host from the network immediately. Conduct a full forensic audit of the host for persistence mechanisms and data exfiltration.'
    },
    'Web Attack': {
        'cve': 'CVE-2022-22965',
        'name': 'Spring4Shell',
        'severity': 9.8,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
        'description': 'RCE in Spring Framework via data binding, targeting web-facing applications.',
        'action': 'Block the source IP via WAF and firewall rules. Review web server logs for signs of webshell deployment or unauthorized file writes.'
    },
    'Bot': {
        'cve': 'CVE-2016-10372',
        'name': 'Mirai Variant',
        'severity': 7.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:H',
        'description': 'Infecting IoT devices to facilitate large-scale DDoS attacks.',
        'action': 'Change all default credentials on network-connected devices. Segment IoT devices onto an isolated VLAN. Block outbound traffic to known botnet C2 IP ranges and enable egress filtering.'
    },
    'FTP-Patator': {
        'cve': 'CVE-2025-49195',
        'name': 'FTP Brute Force (CWE-307)',
        'severity': 5.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N',
        'description': 'Automated authentication attacks against the File Transfer Protocol (FTP). Exploits CWE-307 (Improper Restriction of Excessive Authentication Attempts) due to a lack of login rate-limiting.',
        'action': 'Block the source IP and implement FTP login rate-limiting. Enforce strong password policies on all FTP accounts and review transfer logs for any successful unauthorized access.'
    },
    'SSH-Patator': {
        'cve': 'CVE-2020-1616',
        'name': 'SSH Brute Force (CWE-307)',
        'severity': 5.3,
        'vector': 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:N',
        'description': 'Automated authentication attacks against the Secure Shell protocol. Exploits CWE-307 (Improper Restriction of Excessive Authentication Attempts) to gain unauthorized shell access.',
        'action': 'Block the source IP. Disable SSH password authentication and enforce key-based login only. Restrict SSH access to known IPs via allowlist and audit auth.log for any successful sessions from the attacker. '
    }
}

# --- SESSION STATE INITIALIZATION ---#
# This must be near the top to ensure the "Vault" exists before the app runs
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None


# --- CLOUD OPTIMIZATION CHECK --- #
IS_CLOUD = os.environ.get('RENDER') == 'true' or 'STREAMLIT_SERVER_PORT' in os.environ
MAX_ROWS = 5000 if IS_CLOUD else None

# --- PAGE CONFIGURATION --- #
st.set_page_config(page_title="Vanguard IDS", page_icon="🛡️", layout="wide")

st.markdown("""
    <div style="background-color:#b5cfff; padding:20px; border-radius:10px; border: 1px solid #30363d; border-left: 5px solid #ffffff;">
        <h2 style="color:#ffffff; margin-top:0; font-family: sans-serif;">🛡️ Vanguard: AI-Intrusion Detection System</h2>
        <p style="color:#ffffff; font-size:15px; line-height:1.6;">
            A forensic intelligence tool that utilizes a machine learning engine to bridge the gap between automated detection and human triage 
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
        self.dropout = nn.Dropout(0.3)  #Randomly shuts off 30% of neurons during training to prevent overfitting
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.relu = nn.ReLU() #Activation function to help the model learn non-linear patterns

    # This is the path data takes through the network: input → layer 1 → activation → dropout → layer 2 → activation → output scores
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

    # Loads StandardScaler and LabelEncoder
    scaler = joblib.load(scaler_path)
    le = joblib.load(encoder_path)

    # Read the Strategic Selection Handshake
    if os.path.exists(active_model_info):
        with open(active_model_info, 'r') as f:
            model_type = f.read().strip()
    else:
        model_type = 'pytorch'  # Default fallback

    # Load the engine with the best AUC-ROC
    if model_type == 'sklearn':
        model = joblib.load('models/vanguard_model.pkl')
        is_pytorch = False
    else:
        input_dim = scaler.n_features_in_ #Reads the number of features the scaler was trained on
        num_classes = len(le.classes_)
        model = IDSNetwork(input_dim, num_classes)
        # Load weights into the architecture
        model.load_state_dict(torch.load('models/vanguard_model.pth', map_location=torch.device('cpu'), weights_only=True))
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
    df.columns = df.columns.str.strip() #Removes leading/trailing spaces in column names

    # Optimization for Cloud Deployment
    if IS_CLOUD and len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
        st.warning(f"⚠️ Cloud Optimization: Analyzing {MAX_ROWS} sampled rows.")

    st.subheader("📊 Network Traffic Log Preview")
    st.dataframe(df.head(10))

    if st.button("🔍 Run Forensic Analysis"):
        with st.spinner("Analyzing traffic patterns..."):
            X_input = df.drop(columns=['Label'], errors='ignore') #Removes the label column if it exists and ignores the column isn't there.
            X_input = X_input.select_dtypes(include=[np.number])

            # Infinite values are replaced with NaN and then filled with 0 so they don't cause errors.
            X_input.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_input.fillna(0, inplace=True)

            # Aligns uploaded CSV columns to the scaler
            expected_cols = scaler.feature_names_in_
            X_input = X_input.reindex(columns=expected_cols, fill_value=0)

            #Normalises the data using the same scale learned during training
            X_scaled = scaler.transform(X_input)

            # Perform Inference
            if is_pytorch:
                input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    logits = model(input_tensor)
                    predictions = torch.argmax(logits, dim=1).numpy() #For each row, picks the class with the highest output score as the prediction and converts it from a PyTorch tensor to a NumPy array depending on the engine selected
            else:
                predictions = model.predict(X_scaled)

            #Converts the numeric predictions back into text
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
        st.write(res_df['Threat_Type'].value_counts()) #Counts how many times each threat label appears in the results

    with col2:
        st.write("#### Malicious Traffic Distribution")

        # Filter for attacks
        malicious_df = res_df[res_df['Threat_Type'] != 'BENIGN'] #Filters out normal traffic and displays only rows classified as attacks for the charts and analysis

        # Get counts and transform into a DataFrame - .reset_index() turns the Threat Names into a real column
        attack_counts = malicious_df['Threat_Type'].value_counts().reset_index()

        # Explicitly name the columns for the chart to reference
        attack_counts.columns = ['Threat Type', 'Count']

        # Checks if the dataframe is not empty
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
    # Non-Technical Risk Translator
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

    with st.expander("📝 View Non-Technical Executive Summary"):
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

        # Vector Key
        with st.expander("🗝️ CVSS v3.1 Vector Key (Forensic Reference)"):
            st.markdown("""
            | Metric | Code | Description |
            | :--- | :--- | :--- |
            | **Attack Vector (AV)** | **N** / **A** / **L** / **P** | Network / Adjacent / Local / Physical |
            | **Attack Complexity (AC)** | **L** / **H** | **Low** (Easy to exploit) / **High** (Requires specialized conditions) |
            | **Privileges Required (PR)** | **N** / **L** / **H** | **None** / **Low** (User) / **High** (Admin) |
            | **User Interaction (UI)** | **N** / **R** | **None** (Silent) / **Required** (Victim must click/act) |
            | **Scope (S)** | **U** / **C** | **Unchanged** / **Changed** (Impact spreads to other systems) |
            | **Impact (C/I/A)** | **N** / **L** / **H** | **None** / **Low** / **High** (Confidentiality, Integrity, Availability) |
            """)
            st.caption("Standardized according to the FIRST.org CVSS v3.1 Specification.")

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

                    st.warning( f"**Action Plan:** {intel['action']}")

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