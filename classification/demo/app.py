import random
from pathlib import Path

import torch
import timm
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

import streamlit as st
from streamlit_image_select import image_select

APP_TITLE = "Демо классификации изображений"
PAGE_ICON = "🧠"
CHOICE_TITLES = {
    "Animals": "Классификация животных",
    "FaceEmotions": "Распознавание эмоций по лицу",
}

def open_examples(path, seed=None, sample_size=None):
    data = []
    for p in path.iterdir():
        label = p.name
        img_dir = path / label
        for img in img_dir.iterdir():
            data.append((img, label))
    if sample_size is not None and data:
        rng = random.Random(seed)
        data = [rng.choice(data) for _ in range(sample_size)]
    return data

def load_ckpt(choice, root_dir):
    root_dir = root_dir.resolve().parent.parent / 'models' 
    ckpt_path = root_dir / f"{choice}.ckpt"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model_prefixes = ("model.", "student_model.")
    state_dict = {
        k.split(".", 1)[1]: v
        for k, v in state_dict.items()
        if k.startswith(model_prefixes)
    }

    if choice == "Animals":
        model = timm.create_model('convnextv2_base', pretrained=True, num_classes=3)
        
    elif choice == "FaceEmotions":
        model = timm.create_model(
            'convnextv2_tiny',
            pretrained=True,
            num_classes=7,
            in_chans=1
        )
    model = model.to('cpu')

    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_source, model):
    in_channels = model.stem[0].in_channels
    image_mode = "L" if in_channels == 1 else "RGB"
    image = Image.open(image_source).convert(image_mode)
    data_config = resolve_model_data_config(model)
    if in_channels == 1:
        data_config["input_size"] = (1, *data_config["input_size"][1:])
        if len(data_config.get("mean", ())) != 1:
            data_config["mean"] = (data_config["mean"][0],)
        if len(data_config.get("std", ())) != 1:
            data_config["std"] = (data_config["std"][0],)
    transform = create_transform(**data_config, is_training=False)
    image_tensor = transform(image).unsqueeze(0).to("cpu")
    return image_tensor

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout='wide'
)
seed = int(st.session_state.get("seed", 1))

st.title(APP_TITLE)
choice = st.pills('Выберите классификатор', ["Animals", "FaceEmotions"], default="Animals")
st.caption(CHOICE_TITLES.get(choice, "Интерактивная демонстрация модели классификации"))

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples" / choice
EXAMPLES = open_examples(EXAMPLES_DIR, seed=seed, sample_size=6)


if not EXAMPLES:
    st.error(f"В папке нет изображений: {EXAMPLES_DIR}")
    st.stop()

images, labels = zip(*EXAMPLES)
class_names = sorted([path.name for path in EXAMPLES_DIR.iterdir() if path.is_dir()])

col1, col2 = st.columns([1, 2])
with col1:
    image_placeholder = st.empty()

gallery_placeholder = st.empty()
with col2.container():
    label_probs = st.columns([1 for _ in range(len(class_names))])
    label_metrics = [col.empty() for col in label_probs]
        
    st.number_input("Seed", min_value=0, value=seed, step=1, key="seed")
    image = image_select(
        label="Выберите изображение:",
        images=images,
        captions=labels,
        index=0,
        use_container_width=True,
        key=f"image_select_{choice}_{seed}",
    )
    uploaded_file = st.file_uploader(
        "Или загрузите свое",
        type=["jpg", "jpeg", "png"],
        key=f"uploaded_file_{choice}",
    )
    

current_image = uploaded_file if uploaded_file is not None else image
if current_image is None:
    current_image = images[0]

with st.spinner("Выполняется классификация..."):
    model = load_ckpt(choice, EXAMPLES_DIR)
    current_image_tensor = preprocess_image(current_image, model)
    with torch.no_grad():
        probabilities = torch.softmax(model(current_image_tensor), dim=1).squeeze(0)


class_names, probabilities = zip(*sorted(
    [(cn, p) for cn, p in zip(class_names, probabilities)],
    key=lambda x: x[1],
    reverse=True
))
for i, label in enumerate(class_names):
    prob = float(probabilities[i].item()) if i < len(probabilities) else 0.0
    label_metrics[i].metric(label, f"{prob:.1%}", border=True)

image_placeholder.image(current_image, width=1024)
