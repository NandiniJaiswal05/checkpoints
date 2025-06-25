import streamlit as st
import os
import requests
import torch

MODEL_URL = "https://drive.google.com/uc?export=download&id=1AbCxyz1234567890"
MODEL_PATH = "checkpoints.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

st.title("PyTorch Checkpoint Inference App")
st.write("Your model is loaded!")
