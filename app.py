import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests

@st.cache_resource
def load_generator():
    url = "https://huggingface.co/nandinijaiswal05/Satellite_to_roadmap/resolve/main/checkpoints.pth"
    output = "checkpoints.pth"

    # Download from Hugging Face if not already present
    if not os.path.exists(output):
        st.info("📥 Downloading model from Hugging Face...")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            with requests.get(url, headers=headers, stream=True) as r:
                if r.status_code != 200 or "html" in r.headers.get("Content-Type", ""):
                    st.error("❌ File download failed or is not a valid .pth file.")
                    st.stop()
                with open(output, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Download failed: {e}")
            st.stop()

    # Load U-Net model architecture
    try:
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=3,
            out_channels=3,
            init_features=64,
            pretrained=False,
            trust_repo=True
        )
    except Exception as e:
        st.error(f"❌ Failed to load U-Net architecture: {e}")
        st.stop()

    # Load model weights
    try:
        checkpoint = torch.load(output, map_location='cpu')
        model.load_state_dict(checkpoint)  # ✅ Using plain state_dict
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        st.stop()

    return model

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

# Streamlit UI
st.title("🛰️ Satellite to Roadmap Generator")

uploaded_file = st.file_uploader("📤 Upload a satellite image (side-by-side)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        w, h = image.size
        satellite = image.crop((0, 0, w // 2, h))

        st.subheader("🧭 Satellite Input (Left Half)")
        st.image(satellite, use_container_width=True)

        input_tensor = transform(satellite).unsqueeze(0)

        generator = load_generator()

        with torch.no_grad():
            output = generator(input_tensor)

        roadmap = tensor_to_pil(output)

        st.subheader("🗺️ Predicted Roadmap (Right Output)")
        st.image(roadmap, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error processing image or prediction: {e}")
