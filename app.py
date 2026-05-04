import sys
sys.path.append("src")

import os
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import get_model
import base64
from io import BytesIO

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Pokemon Classifier",
    page_icon="🎮",
    layout="wide"
)

# =========================
# Restart State
# =========================
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# =========================
# CSS
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #222222;
    transform: translateY(-40px);
}

[data-testid="stHeader"] {
    background-color: #ffffff;
}

body {
    background-color: #f6f7f9;
}

/* 상단바 */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 40px;
    background-color: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 40px;
}

.logo {
    font-size: 22px;
    font-weight: 800;
}

.menu {
    font-size: 22px;
}

/* restart */
.restart-icon {
    text-decoration: none;
    font-size: 24px;
    color: #222222;
    cursor: pointer;
    transition: transform 0.2s ease;
}

.restart-icon:hover {
    transform: rotate(25deg);
}

/* 왼쪽 이미지 */
.left-box {
    text-align: center;
}

/* uploader 카드 */
div[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    height: 420px;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* label 숨김 */
div[data-testid="stFileUploader"] > label {
    display: none;
}

/* 내부 영역 */
div[data-testid="stFileUploader"] section {
    width: 260px !important;
    height: 72px !important;
    min-height: 72px !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* drag 안내 제거 */
div[data-testid="stFileUploader"] section > div {
    display: none !important;
}

/* 버튼 스타일 */
div[data-testid="stFileUploader"] button {
    background-color: #1f6feb !important;
    color: white !important;
    padding: 16px 34px !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    border: none !important;
    font-size: 16px !important;
}

/* 업로드 후 미리보기 카드 */
.preview-card {
    background: #ffffff;
    border-radius: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    height: 420px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
    margin-top: -20px;
}

.preview-card img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 15px;
}

/* 결과 카드 */
.result-card {
    margin-top: 28px;
    padding: 24px;
    border-radius: 20px;
    background: #fff;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    text-align: center;
}

.pred-name {
    font-size: 30px;
    font-weight: 800;
    color: #ff4d4f;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Top Bar
# =========================
st.markdown("""
<div class="topbar">
    <div class="logo">pokemon classifier</div>
    <a class="restart-icon" href="?restart=1">🔄</a>
</div>
""", unsafe_allow_html=True)

if st.query_params.get("restart") == "1":
    st.session_state["uploader_key"] += 1
    st.query_params.clear()
    st.rerun()

# =========================
# Settings
# =========================
MODEL_PATH = "models/resnet50_full.pth"
MODEL_NAME = "resnet50"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

num_classes = len(class_names)

# =========================
# Model
# =========================
@st.cache_resource
def load_model():
    model = get_model(
        name=MODEL_NAME,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# Transform
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, k=5)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        results.append({
            "Pokemon": class_names[idx.item()],
            "Probability": float(prob.item())
        })

    return results

# =========================
# 이미지 → base64 변환
# =========================
def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# =========================
# Layout
# =========================
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.image("assets/main.jpg", use_container_width=True)

with col2:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_base64 = image_to_base64(image)

        # uploader 숨기고 이미지로 대체
        st.markdown(f"""
        <style>
        div[data-testid="stFileUploader"] {{
            display: none !important;
        }}
        </style>

        <div class="preview-card">
            <img src="data:image/png;base64,{img_base64}">
        </div>
        """, unsafe_allow_html=True)

# =========================
# Result
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    results = predict(image)
    top1 = results[0]

    st.markdown(f"""
    <div class="result-card">
        <div class="pred-name">{top1["Pokemon"]}</div>
        <div>{top1["Probability"] * 100:.2f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    for r in results:
        st.write(f"{r['Pokemon']} - {r['Probability'] * 100:.2f}%")