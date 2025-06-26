import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import functools

# Minimal ResNetBlock & Generator
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            norm_layer(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super().__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf*2, 3, 2, 1),
            norm_layer(ngf*2),
            nn.ReLU(True),

            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1),
            norm_layer(ngf*4),
            nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model.append(ResnetBlock(ngf*4, norm_layer))

        model += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1),
            norm_layer(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1),
            norm_layer(ngf),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = ResnetGenerator(3, 3)
    model.load_state_dict(torch.load("latest_net_G.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)

def postprocess(tensor):
    tensor = tensor.squeeze().detach().cpu()
    tensor = (tensor + 1) / 2
    return transforms.ToPILImage()(tensor)

st.title("🛰️ Satellite to Roadmap Converter")

file = st.file_uploader("Upload a Satellite Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Input Image", use_column_width=True)

    model = load_model()
    input_tensor = preprocess(image)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = postprocess(output_tensor)
    st.image(output_image, caption="Output Roadmap Image", use_column_width=True)
