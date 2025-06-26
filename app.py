import streamlit as st
from PIL import Image
import torch
import os
import requests
import torchvision.transforms as transforms

@st.cache_resource
def load_generator():
    # 🔁 Replace this with your correct Dropbox direct link
    url = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
    output_path = "checkpoints.pth"

    if not os.path.exists(output_path):
        st.info("📥 Downloading model from Dropbox...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

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
        st.error(f"❌ Failed to load model architecture: {e}")
        st.stop()

    try:
        checkpoint = torch.load(output_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'gen_model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['gen_model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load weights: {e}")
        st.stop()

    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

# UI
st.title("🛰️ Satellite to Roadmap Generator")

uploaded_file = st.file_uploader("📤 Upload a satellite image (side-by-side)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Split left half
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
        st.error(f"❌ Error processing image or generating roadmap: {e}")
