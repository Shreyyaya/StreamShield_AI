import streamlit as st
import requests #calls fastapi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess_values(values):
    protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}

    flag_map = {
        'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3,
        'SH': 4, 'RSTO': 5, 'S1': 6, 'S2': 7, 'S3': 8
    }

    # Basic service hashing
    def encode_service(val):
        return float(abs(hash(val)) % 50)  #range 0-49

    processed = []

    for i, v in enumerate(values):
        v = str(v).strip()

        try:
            processed.append(float(v))
        except:
            if v in protocol_map:
                processed.append(protocol_map[v])
            elif v in flag_map:
                processed.append(flag_map[v])
            else:
                # treat as service or unknown categorical
                processed.append(encode_service(v))

    return processed

st.set_page_config(page_title="StreamShield AI", layout="wide")

API_BASE = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_BASE}/predict"
METRICS_URL = f"{API_BASE}/metrics"

NUM_FEATURES = 41

st.title("StreamShield AI Dashboard")

# -------------------------
# API STATUS
# -------------------------
def check_api():
    try:
        r = requests.get(API_BASE, timeout=2)
        return r.status_code == 200
    except:
        return False

if check_api():
    st.sidebar.success("API Running")
else:
    st.sidebar.error("API Not Running")

# -------------------------
# SESSION STATE
# -------------------------
for i in range(NUM_FEATURES):
    if f"f_{i}" not in st.session_state:
        st.session_state[f"f_{i}"] = 0.0

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox("Model", ["Model A", "Model B"])

# Auto fill
if st.sidebar.button("Auto Fill"):
    for i in range(NUM_FEATURES):
        if i == 1:  # protocol
            st.session_state[f"f_{i}"] = np.random.choice([0,1,2])
        elif i == 2:  # service
            st.session_state[f"f_{i}"] = np.random.randint(0,50)
        else: #random 
            st.session_state[f"f_{i}"] = float(np.random.randint(0, 300))

    st.rerun() #refreshes with new values

# CSV Upload

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    # Clean NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    values = df.iloc[0].tolist()[:41]   

    processed = preprocess_values(values)

    for i in range(min(NUM_FEATURES, len(processed))):
        st.session_state[f"f_{i}"] = processed[i]  #stores values

# TEXT INPUT (comma separated)
text_input = st.sidebar.text_area("Paste comma-separated values")

if st.sidebar.button("Load Text Input"):
    try:
        raw_values = text_input.strip().split(",")
        processed = preprocess_values(raw_values)

        for i in range(min(NUM_FEATURES, len(processed))):
            st.session_state[f"f_{i}"] = processed[i]

    except Exception as e:
        st.sidebar.error(f"Invalid input: {e}")

# -------------------------
# FEATURES GRID
# -------------------------
st.subheader("Features")

cols = st.columns(4)

for i in range(NUM_FEATURES):
    with cols[i % 4]:
        st.number_input(
            f"F{i+1}",
            key=f"f_{i}",
            step=0.1
        )

# -------------------------
# PREDICT
# -------------------------
if st.button("Predict"):
    try:
        features = []

        for i in range(NUM_FEATURES):
            val = st.session_state[f"f_{i}"]

            # Safe conversion
            try:
                val = float(val)
            except:
                val = 0.0

            if np.isnan(val) or np.isinf(val):
                val = 0.0

            features.append(val)

        if all(v == 0 for v in features):
            st.error("All features are zero → bad preprocessing")
            st.stop()

        st.write("Features:", features)

        res = requests.post(PREDICT_URL, json={
            "features": features,
            "model": model_choice
        })

        if res.status_code != 200:
            st.error(res.text)
            st.stop()

        data = res.json()

        st.subheader("Prediction")
        st.write("Result:", data.get("result"))
        st.write("Class ID:", data.get("prediction"))

    except Exception as e:
        st.error(f"Error: {e}")

    # -------------------------
    # ALWAYS SHOW GRAPH
    # -------------------------
    st.subheader("Feature Graph")

    fig, ax = plt.subplots()
    ax.bar(range(NUM_FEATURES), features)
    ax.set_title("Feature Distribution")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Value")

    st.pyplot(fig)

# -------------------------
# METRICS SECTION
# -------------------------
st.subheader("Model Metrics")

if st.button("Load Metrics"):
    try:
        res = requests.get(METRICS_URL)
        data = res.json()

        st.write("Accuracy:", data.get("accuracy"))

        if "confusion_matrix" in data:
            cm = np.array(data["confusion_matrix"])

            fig, ax = plt.subplots()
            ax.imshow(cm)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j],
                            ha="center", va="center")

            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    except Exception as e:
        st.error(e)