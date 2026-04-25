import time
import hashlib
import pathlib
import base64
import io
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter


import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training import SelfMadeUNet

if "best_time_models" not in st.session_state:
    st.session_state.best_time_models = {}
if "best_miou_models" not in st.session_state:
    st.session_state.best_miou_models = {}
if "best_per_iou_models" not in st.session_state:
    st.session_state.best_per_iou_models = {}
if "prediction_cache" not in st.session_state:
    st.session_state.prediction_cache = {}
if "compare_cache" not in st.session_state:
    st.session_state.compare_cache = {}

st.set_page_config(
    page_title="Сравнение моделей семантической сегментации",
    page_icon="🧩",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://docs.streamlit.io/",
        "About": """
    ### Сравнение моделей семантической сегментации
    Сайт для интерактивного сравнения моделей: Unet (EffNet/VGG), DeepLabV3+, SegFormer.  
    Поддерживается загрузка своего изображения и сравнение с GT-маской на примерах (mIoU, IoU по животным, время инференса).
    """
        },
    )
st.header("Сравнение моделей семантической сегментации 🧩")

colors_map = {
  0:  '#FF69B4',
  1:  '#FF7F00',
  2:  '#804080',
  3:  '#DC143C',
  4:  '#FFFFFF',
  5:  '#6B8E23',
  6:  '#1E90FF',
  7:  '#FF4500',
  8:  '#4682B4',
  9:  '#464646',
  10: '#999999',
  11: '#FFD700',
  12: '#ADFF2F',
  13: '#00008E',
  14: '#000000',
}

COMPARE_MASK_LEGEND = {
  0: ('Расхождение масок', [200, 0, 0]),
  1: ('Совпадение масок', [0, 200, 0]),
  2: ('Игнорируемая зона (void)', [40, 40, 40]),
}

class_names = {
  0:  'Животные',
  1:  'Преграды',
  2:  'Плоскости (дорога)',
  3:  'Люди',
  4:  'Разметка',
  5:  'Природа',
  6:  'Объекты',
  7:  'Баннеры',
  8:  'Небо',
  9:  'Здания',
  10: 'Поддержки',
  11: 'Светофоры',
  12: 'Дорожный знак',
  13: 'Транспортные средства',
  14: 'Пустота',
}

best_time_models = st.session_state.best_time_models
best_miou_models = st.session_state.best_miou_models
best_per_iou_models = st.session_state.best_per_iou_models

UPLOAD_HEIGHT = 512
UPLOAD_WIDTH = 768

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
    

def _image_cache_key(image):
    hasher = hashlib.sha256()
    hasher.update(image.mode.encode("ascii", "ignore"))
    hasher.update(f"{image.size[0]}x{image.size[1]}".encode("ascii"))
    hasher.update(image.tobytes())
    return hasher.hexdigest()

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
JX_JS_PATH = ASSETS_DIR / "juxtapose.min.js"
JX_CSS_PATH = ASSETS_DIR / "juxtapose.css"

@st.cache_data(show_spinner=False)
def _load_asset_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def _read_image_as_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.shape[0] < 5:
            image = image[:, :, ::-1]
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError("Unsupported image type for comparison")

def _pillow_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", subsampling=0, quality=95)
    image_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{image_str}"

def image_comparison_local(
    img1,
    img2,
    label1: str = "1",
    label2: str = "2",
    width: int = 704,
    show_labels: bool = True,
    starting_position: int = 50,
    make_responsive: bool = True,
):
    css = _load_asset_text(JX_CSS_PATH)
    js = _load_asset_text(JX_JS_PATH)
    if not css or not js:
        stop_with_error("Missing local juxtapose assets. Ensure assets/juxtapose.* exist.")

    img1_pil = _read_image_as_pil(img1)
    img2_pil = _read_image_as_pil(img2)
    img_width, img_height = img1_pil.size
    h_to_w = img_height / img_width
    height = int((width * h_to_w) * 0.95)

    img1_b64 = _pillow_to_base64(img1_pil)
    img2_b64 = _pillow_to_base64(img2_pil)

    width_style = f"{width}px" if width else "100%"
    htmlcode = f"""
        <style>body {{ margin: 0; }}</style>
        <style>{css}</style>
        <div id="jx-container" style="height: {height}px; width: {width_style};"></div>
        <script>{js}</script>
        <script>
        const slider = new juxtapose.JXSlider('#jx-container',
            [
                {{
                    src: '{img1_b64}',
                    label: '{label1}',
                }},
                {{
                    src: '{img2_b64}',
                    label: '{label2}',
                }}
            ],
            {{
                animate: true,
                showLabels: {'true' if show_labels else 'false'},
                showCredits: true,
                startingPosition: "{starting_position}%",
                makeResponsive: {'true' if make_responsive else 'false'},
            }});
        </script>
        """
    return components.html(htmlcode, height=height, width=width)

def _align_orig_mask(orig_mask, target_shape):
    if orig_mask is None:
        return None
    if not torch.is_tensor(orig_mask):
        orig_mask = torch.as_tensor(orig_mask)
    if orig_mask.ndim == 3 and orig_mask.shape[0] == 1:
        orig_mask = orig_mask.squeeze(0)
    if orig_mask.ndim != 2:
        return orig_mask
    target_h, target_w = target_shape
    if orig_mask.shape[0] != target_h or orig_mask.shape[1] != target_w:
        orig_mask = orig_mask.unsqueeze(0).unsqueeze(0).float()
        orig_mask = F.interpolate(orig_mask, size=(target_h, target_w), mode="nearest")
        orig_mask = orig_mask.squeeze(0).squeeze(0).long()
    return orig_mask

def _get_cached_prediction(image):
    global selected_model
    cache = st.session_state.prediction_cache
    image_key = _image_cache_key(image)
    cache_key = (selected_model, image_key)
    cached = cache.get(cache_key)
    if cached:
        return cached["mask"], cached["inf_time"], cached["num_classes"], image_key, True

    transform = transforms.Compose([
        transforms.Resize((512, 768)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to("cpu")
    with st.spinner("Запущен инференс (CPU)..."):
        start_time = time.perf_counter()
        with torch.no_grad():
            pred = model(image_tensor).squeeze(0)
        inf_time = time.perf_counter() - start_time
    num_classes = pred.shape[0]
    mask = pred.argmax(dim=0).cpu().numpy().astype(np.uint8)

    cache[cache_key] = {
        "mask": mask,
        "inf_time": inf_time,
        "num_classes": num_classes,
    }

    return mask, inf_time, num_classes, image_key, False

# @st.cache_data
def prediction(image, orig_mask=None):
    global selected_model, arg_example
    mask, inf_time, num_classes, image_key, _ = _get_cached_prediction(image)

    best_time_models.setdefault(arg_example, {})
    best_time_models[arg_example].setdefault('time', 100)
    best_time_models[arg_example].setdefault('name', '')
    diff_time = best_time_models[arg_example]['time'] - inf_time
    if best_time_models[arg_example]['time'] > inf_time:
        best_time_models[arg_example]['time'] = inf_time
        best_time_models[arg_example]['name'] = selected_model

    col1, col2, col3 = st.columns(3)
    if diff_time < 90:
        model_name = best_time_models[arg_example]['name']
        col1.metric('Время предсказания (CPU)', f'{inf_time:.2f} сек.',  f'{diff_time:.2f} | ...{model_name[-10:]}')
    else:
        col1.metric('Время предсказания (CPU)', f'{inf_time:.2f} сек.')

    if orig_mask != None:
        orig_mask = _align_orig_mask(orig_mask, mask.shape[:2])
        orig_mask_np = orig_mask.cpu().numpy() if torch.is_tensor(orig_mask) else np.asarray(orig_mask)
        compare_cache = st.session_state.compare_cache
        compare_key = (selected_model, image_key)
        cached_compare = compare_cache.get(compare_key)
        if cached_compare:
            compare_mask = cached_compare["compare_mask"]
            miou = cached_compare["miou"]
            animal_iou = cached_compare["animal_iou"]
        else:
            compare_mask = np.zeros_like(mask)
            compare_mask[mask == orig_mask_np] = 1
            compare_mask[orig_mask_np == 14] = 2 

            tp, fp, fn, tn = smp.metrics.get_stats(torch.from_numpy(mask), orig_mask,
                                               mode="multiclass",
                                               num_classes=14,
                                               ignore_index=14)

            iou = smp.metrics.iou_score(tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0))
            animal_iou = iou[0]
            miou = iou.mean()
            compare_cache[compare_key] = {
                "compare_mask": compare_mask,
                "miou": miou,
                "animal_iou": animal_iou,
            }

        mask = compare_mask
        palette = np.array(
            [COMPARE_MASK_LEGEND[i][1] for i in sorted(COMPARE_MASK_LEGEND)],
            dtype=np.uint8,
        )
        
        best_miou_models.setdefault(arg_example, {})
        best_per_iou_models.setdefault(arg_example, {})
        best_miou_models[arg_example].setdefault('miou', 0)
        best_per_iou_models[arg_example].setdefault('per_iou', 0)
        best_miou_models[arg_example].setdefault('name', '')
        best_per_iou_models[arg_example].setdefault('name', '')

        diff_miou = miou - best_miou_models[arg_example]['miou']
        diff_per_iou = animal_iou - best_per_iou_models[arg_example]['per_iou']

        if best_miou_models[arg_example]['miou'] < miou:
            best_miou_models[arg_example]['miou'] = miou
            best_miou_models[arg_example]['name'] = selected_model
        if best_per_iou_models[arg_example]['per_iou'] < animal_iou:
            best_per_iou_models[arg_example]['per_iou'] = animal_iou
            best_per_iou_models[arg_example]['name'] = selected_model


        if diff_miou != miou:
            model_name = best_miou_models[arg_example]['name']
            model_per_iou_name = best_per_iou_models[arg_example]['name']
            col2.metric('Cредний mIoU', f'{miou:.3f}', f'{diff_miou:.3f} | ...{model_name[-14:]}')
            col3.metric('IoU по животным', f'{animal_iou:.3f}', f'{diff_per_iou:.3f} | ...{model_per_iou_name[-14:]}')
        else:            
            col2.metric('Cредний mIoU', f'{miou:.3f}')
            col3.metric('IoU по животным', f'{animal_iou:.3f}')
        
    else:
        
        palette = np.array([
            hex_to_rgb(colors_map.get(i, "#000000"))
            for i in range(num_classes)
        ], dtype=np.uint8)

    mask_rgb = palette[mask]
    mask_rgb_pil = Image.fromarray(mask_rgb).convert("RGBA").resize(image.size, Image.NEAREST)       

    return mask_rgb_pil



def visualisation():          
    global image, orig_mask
    compare_mask = False    

    if orig_mask != None:
        compare_mask = st.toggle('Сравнить маски')    
    
    
    if compare_mask:    
        mask_rgb_pil = prediction(image, orig_mask)
        with st.sidebar:
            alpha = st.slider("Альфа", 0.0, 1.0, 0.5, 0.05)
    else:
        mask_rgb_pil = prediction(image)
        with st.sidebar:
            alpha = st.slider("Альфа", 0.0, 1.0, 0.7, 0.05)  
    

    
    overlay = Image.blend(image.convert("RGBA"), mask_rgb_pil, alpha)
  
    image_comparison_local(
        img1=image,
        img2=overlay,
        label1="Оригинал",
        label2="Сегментация",
        width=700,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
    )

    legend(compare_mask)


def legend(compare_mask):
    legend_box = st.sidebar.container()

    if compare_mask:
        with legend_box:
            st.markdown("### Легенда сравнения")
            st.caption("Цвета показывают, где предсказанная маска совпадает с эталонной.")
            for _, (label, value) in COMPARE_MASK_LEGEND.items():
                r, g, b = value
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <div style="width:16px;height:16px;background:rgb({r},{g},{b});
                                    border:1px solid #555;flex:0 0 16px;"></div>
                        {label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        with legend_box:
            st.markdown("### Легенда")
            for i in range(len(class_names)):
                c = colors_map.get(i, "#000000")
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <div style="width:16px;height:16px;background:{c};
                                    border:1px solid #555;flex:0 0 16px;"></div>
                        {class_names[i]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def select_model(model_name):
    if model_name.startswith('UNET_VGG-19'):
        backbone = torchvision.models.vgg19_bn(weights=None, progress=False).to('cpu')
        model = SelfMadeUNet(14, backbone).to('cpu')
    elif model_name.startswith('U'):
        model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    elif model_name.startswith('DEEPLAB'):
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    else:
        model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )
    return model

def stop_with_error(message):
    st.error(message)
    st.stop()

@st.cache_resource(show_spinner=True)
def get_model(selected_model):
    checkpoint_path = Path(parent_path) / model_names[selected_model]
    if not checkpoint_path.exists():
        stop_with_error(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception:
        stop_with_error(f"Failed to load checkpoint: {checkpoint_path}")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        model_state_dict = checkpoint
    else:
        stop_with_error(f"Unexpected checkpoint format: {checkpoint_path}")

    model = select_model(selected_model)
    model = Model(model)
    try:
        model.load_state_dict(model_state_dict)
    except Exception as exc:
        stop_with_error(f"Failed to load model weights: {checkpoint_path}\n{exc}")
    model.eval()

    return model


def hex_to_rgb(hex_color):
    hex_color = hex_color.partition('#')[2]
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)







parent_path = '../models/'
model_names = {    
    'UNET_EFFNET-B3': 'unet_effnet_ignore-epoch=39.ckpt',
    'UNET_VGG-19': 'unet_vgg19_ignore-epoch=25.ckpt',
    'SEGFORMER_MIT-B2': 'segformer_ignore-epoch=46.ckpt',    
    'DEEPLAB_V3+_RESNET50': 'deeplab_v3_plus_ignore-epoch=42.ckpt',      
    'SF_MIT-B2_ENLARGED': 'segformer_enlarge-epoch=46.ckpt',
}

with st.sidebar:
    selected_model = st.selectbox('Сегментационные модели',
                                  [name for name in model_names])
model = get_model(selected_model)

with st.sidebar:
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((UPLOAD_WIDTH, UPLOAD_HEIGHT), Image.BILINEAR)
        except Exception:
            stop_with_error("Failed to read uploaded image.")
        arg_example = 0
        orig_mask = None
    else:
        examples_path = Path('examples/')
        if not examples_path.exists():
            stop_with_error(f"Examples folder not found: {examples_path}")
        examples = [f for f in examples_path.iterdir() if f.suffix != '.pt' and f.is_file()]
        if not examples:
            stop_with_error(f"No example images found in: {examples_path}")
        arg_example = st.pills('Примеры', [i + 1 for i in range(len(examples))], default=1)
        if arg_example is None:
            st.stop()
        try:
            image = Image.open(examples[arg_example - 1]).convert("RGB")
        except Exception:
            stop_with_error(f"Failed to read example image: {examples[arg_example - 1]}")
        mask_path = examples[arg_example - 1].with_suffix('.jpg.pt')
        if not mask_path.exists():
            stop_with_error(f"Mask file not found: {mask_path}")
        try:
            orig_mask = torch.load(mask_path, map_location=torch.device('cpu'))
        except Exception:
            stop_with_error(f"Failed to load mask file: {mask_path}")



visualisation()
