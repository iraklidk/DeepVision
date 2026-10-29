from pathlib import Path

import torch
import streamlit as st
from PIL import Image

from src.data import make_transforms
from src.model import build_model

RUN_NAME = "baseline"
CHECKPOINT_PATH = Path("outputs") / RUN_NAME / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_classifier():
    if not CHECKPOINT_PATH.exists():
        return None
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model, _ = build_model(checkpoint.get("unfreeze_last_block", False))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    _, transform = make_transforms()
    return model, transform


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱")
st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image and the ResNet-18 classifier will predict whether it shows a cat or a dog.")

classifier = load_classifier()
if classifier is None:
    st.warning(
        "No trained model was found. First run `python train.py --run-name baseline`, "
        "then refresh this page."
    )
    st.stop()

model, transform = classifier
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax().item())
    class_names = ["Cat", "Dog"]
    st.subheader(f"Prediction: {class_names[predicted_index]}")
    st.write(f"Cat score: {probabilities[0].item():.1%}")
    st.write(f"Dog score: {probabilities[1].item():.1%}")
    st.caption("Scores are model estimates, not guaranteed certainty.")
