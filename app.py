import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from cycle_gan_model import CycleGANModel

# Minimal opt class to mimic training args
class Opt:
    def __init__(self):
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.netG = 'resnet_9blocks'
        self.norm = 'instance'
        self.no_dropout = True
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.direction = 'AtoB'
        self.gpu_ids = []
        self.isTrain = False
        self.checkpoints_dir = '.'
        self.name = 'inference'
        self.preprocess = 'resize'
        self.epoch = 'latest'
        self.load_iter = 0
        self.continue_train = False
        self.verbose = False

@st.cache_resource
def load_model():
    opt = Opt()
    model = CycleGANModel(opt)
    model.setup(opt)
    state_dict = torch.load("latest_net_G.pth", map_location="cpu")
    model.netG_A.load_state_dict(state_dict)
    model.netG_A.eval()
    return model.netG_A

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

# === Streamlit UI ===
st.title("🛰️ Satellite to Roadmap Converter (CycleGAN)")

uploaded = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Satellite Image", use_column_width=True)

    with st.spinner("Translating..."):
        model = load_model()
        input_tensor = preprocess(image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_image = postprocess(output_tensor)

    st.image(output_image, caption="Generated Roadmap Image", use_column_width=True)
