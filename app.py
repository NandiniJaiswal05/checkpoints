import streamlit as st
import os
import requests
import convert_model

MODEL_URL = "https://drive.google.com/file/d/13RiUDLFkhGtO6g1KDS0bdcCMHlSDeY_g/view?usp=sharing"
MODEL_PATH = "checkpoints.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = convert_model.load(MODEL_PATH, map_location=convert_model.device('cpu'))
    model.eval()
    return model

model = load_model()

st.title("PyTorch Checkpoint Inference App")
st.write("Your model is loaded!")
