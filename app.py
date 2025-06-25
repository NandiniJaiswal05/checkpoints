import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import urllib.request

@st.cache_resource
def load_generator():
    url = "https://huggingface.co/nandinijaiswal05/Satellite_to_roadmap/resolve/main/checkpoints.pth"
    output = "checkpoints.pth"

    if not os.path.exists(output):
        st.info("📦 Downloading model from Hugging Face...")
        urllib.request.urlretrieve(url, output)

    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=3,
        out_channels=3,
        init_features=64,
        pretrained=False
    )
    checkpoint = torch.load(output, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['gen_model_state_dict'])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def tensor_to_pil(tensor_img):
    tensor_img = tensor_img.squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(tensor_img)

st.title("🛰️ Satellite to Roadmap Generator")

uploaded_file = st.file_uploader("📤 Upload a satellite-to-roadmap image", type=["jpg", "jpeg", "png"])

if uploaded_file:
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
