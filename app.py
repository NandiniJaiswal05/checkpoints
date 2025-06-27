import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import networks  # Needed to define the generator

from cycle_gan_model import CycleGANModel

# === Minimal options to mimic training-time arguments ===
class InferenceOptions:
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
        self.name = 'latest_net_G'
        self.preprocess = 'resize'
        self.epoch = 'latest'
        self.load_iter = 0
        self.continue_train = False
        self.verbose = False

# === Load the trained generator model ===
@st.cache_resource
def load_model():
    opt = InferenceOptions()
    model = CycleGANModel(opt)

    # Instead of model.setup(opt), define netG_A manually
    model.netG_A = networks.define_G(
        opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids
    )

    state_dict = torch.load("latest_net_G.pth", map_location="cpu")
    model.netG_A.load_state_dict(state_dict)
    model.netG_A.eval()
    return model.netG_A

# === Preprocess uploaded image ===
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# === Postprocess output tensor into PIL image ===
def postprocess_tensor(tensor):
    tensor = tensor.squeeze().detach().cpu()
    tensor = (tensor + 1) / 2
    return transforms.ToPILImage()(tensor)

# === Streamlit UI ===
st.set_page_config(page_title="Satellite to Roadmap Generator", layout="wide")
st.title("🛰️ Satellite to Roadmap Generator")
st.markdown("Convert satellite images into roadmap-style visuals using a pre-trained CycleGAN model.")

uploaded_file = st.file_uploader("📁 Upload a Satellite Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🛰️ Input Satellite Image", use_column_width=True)

    with st.spinner("🔄 Translating to roadmap..."):
        model = load_model()
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_image = postprocess_tensor(output_tensor)

    st.image(output_image, caption="🗺️ Generated Roadmap Image", use_column_width=True)
    st.success("✅ Translation complete!")
