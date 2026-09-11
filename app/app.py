"""
SignalScope — Streamlit app shell.

Right now this calls the PLACEHOLDER predict() function from model/predict.py.
Nothing here needs to change when the real model is plugged in — it will
just start returning real results automatically.

Run with: streamlit run app/app.py   (from the project root)
"""

import sys
import os

# allow importing from the model/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

import streamlit as st
from predict import predict

st.set_page_config(page_title="SignalScope", page_icon="🔍", layout="centered")

st.title("SignalScope")
st.caption("Telling Real From Synthetic in the Age of Generative Media")

uploaded_file = st.file_uploader(
    "Upload an image to check", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    # save to a temp path since predict() expects a file path
    temp_path = os.path.join("temp_upload.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing..."):
        result = predict(temp_path)

    label = result["label"]
    confidence = result["confidence"]

    if label == "fake":
        st.error(f"Likely AI-generated — confidence {confidence:.2f}")
    else:
        st.success(f"Likely real — confidence {confidence:.2f}")

    os.remove(temp_path)
else:
    st.info("Upload an image above to get a prediction.")
