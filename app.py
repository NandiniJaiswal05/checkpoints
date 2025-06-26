import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os
import requests

# --- CycleGAN Generator (ResNet-based) ---
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=False)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf, affine=True, track_running_stats=False),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2, affine=True, track_running_stats=False),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2), affine=True, track_running_stats=False),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# --- Load Generator ---
@st.cache_resource
def load_generator():
    url = "https://www.dropbox.com/scl/fi/wrae5qoxvmc432whdi8fc/checkpoints.pth?rlkey=ilw12iytudgwi1o0ykqd5tdgh&dl=1"
    model_path = "checkpoints.pth"

    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Dropbox...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
    except Exception as e:
        st.error(f"❌ Failed to load model weights: {e}")
        st.stop()

    return model

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_pil(tensor_img):
    img = tensor_img.squeeze().detach().cpu() * 0.5 + 0.5
    return transforms.ToPILImage()(img)

# --- Streamlit UI ---
st.set_page_config(page_title="Satellite to Roadmap - CycleGAN", layout="wide")
st.title("🛰️ Satellite to Roadmap - CycleGAN")

uploaded_file = st.file_uploader("📤 Upload a satellite image (256x256 or larger)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        input_tensor = transform(image).unsqueeze(0)

        generator = load_generator()

        with torch.no_grad():
            output = generator(input_tensor)

        roadmap = tensor_to_pil(output)

        st.subheader("🗺️ Generated Roadmap")
        st.image(roadmap, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
