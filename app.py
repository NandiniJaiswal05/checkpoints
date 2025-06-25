import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import requests

# ==========================
# Load the Generator Model
# ==========================
import streamlit as st
import torch
import os
import requests
import mimetypes

@st.cache_resource
def load_generator():
    url = "https://huggingface.co/nandinijaiswal05/Satellite_to_roadmap/resolve/54ff31a4a0a7d0d8fbb2cdcb45021d302cfb3284/checkpoints.pth"
    output = "checkpoints.pth"

    # Step 1: Download file safely
    if not os.path.exists(output):
        st.info("📥 Downloading model from Hugging Face (via Git LFS)...")
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            with requests.get(url, headers=headers, stream=True) as r:
                content_type = r.headers.get("Content-Type", "")
                if r.status_code != 200 or "html" in content_type.lower():
                    st.error(f"❌ File download failed. Type: {content_type}")
                    st.stop()
                with open(output, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Download error: {e}")
            st.stop()

    # Step 2: Load model architecture (U-Net)
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
        st.error(f"❌ U-Net architecture load failed: {e}")
        st.stop()

    # Step 3: Try loading the .pth file in every known format
    try:
        checkpoint = torch.load(output, map_location='cpu')

        if isinstance(checkpoint, dict):
            if 'gen_model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['gen_model_state_dict'])
                st.success("✅ Loaded from full checkpoint dictionary with 'gen_model_state_dict'")
            elif all(isinstance(k, str) for k in checkpoint.keys()):
                model.load_state_dict(checkpoint)
                st.success("✅ Loaded from plain state_dict")
            else:
                st.error("❌ .pth file contains an unrecognized dict format.")
                st.stop()

        else:
            st.warning("⚠️ The file is not a dict. Attempting TorchScript load (unlikely).")
            try:
                model = torch.jit.load(output, map_location='cpu')
                st.success("✅ Loaded as a TorchScript model (jit)")
            except Exception as e:
                st.error(f"❌ TorchScript load also failed: {e}")
                st.stop()

    except Exception as e:
        st.error(f"❌ torch.load failed: {e}")
        st.stop()

    model.eval()
    return model



# ==========================
# Preprocessing Utilities
# ==========================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

# ==========================
# Streamlit App UI
# ==========================
st.title("🛰️ Satellite to Roadmap Generator")

uploaded_file = st.file_uploader("📤 Upload a satellite image (side-by-side)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Extract left half (satellite input)
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
