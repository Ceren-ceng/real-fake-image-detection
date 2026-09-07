# -*- coding: utf-8 -*-
"""
REAL / FAKE LIVE TEST UYGULAMASI
Final sunum icin canli demo.

Calistirma:
    pip install streamlit
    streamlit run live_test_REAL_FAKE.py

Varsayilan model:
    CNN-v3 Stronger (final/dengeli model)

Istersen arayuzden V4 veya V5 de secilebilir.

NOT:
- Model yeniden egitilmez.
- Kayitli 5 fold BEST_MODEL checkpointleri kullanilir.
- Her fold icin TTA = normal + yatay cevrilmis goruntu.
- 5 fold olasiliklari ortalanir.
"""

from pathlib import Path, PureWindowsPath
import io
import subprocess
import tempfile

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import torch
import torch.nn as nn


# =========================================================
# AYARLAR
# =========================================================

BASE = Path(r"C:\Users\Ceren\Downloads\yapay sinir ağları")

SIZE = 320
THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_DIRS = {
    "V3 Stronger (Final / önerilen)": (
        BASE
        / "cross_validation_master408"
        / "cnn_v3_stronger_master408"
    ),
    "V4 MultiScale512": (
        BASE
        / "cross_validation_master408"
        / "cnn_v4_multiscale512"
    ),
    "V5 Extended384": (
        BASE
        / "cross_validation_master408"
        / "cnn_v5_extended384"
    ),
}

HEIC_EXTS = {".heic", ".heif"}


# =========================================================
# HEIC
# =========================================================

def find_imagemagick():
    candidates = ["magick", "convert"]

    for cmd in candidates:
        try:
            result = subprocess.run(
                [cmd, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            if result.returncode == 0:
                return cmd

        except FileNotFoundError:
            pass

    return None


IMAGEMAGICK = find_imagemagick()


def decode_heic_bytes(data, suffix=".heic"):
    if IMAGEMAGICK is None:
        raise RuntimeError(
            "HEIC icin ImageMagick bulunamadi. "
            "Canli sunumda JPEG/PNG kullanabilirsin."
        )

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as tmp:
        tmp.write(data)
        temp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [
                IMAGEMAGICK,
                str(temp_path),
                "png:-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "HEIC goruntu acilamadi:\n"
                + result.stderr.decode(
                    "utf-8",
                    errors="replace"
                )
            )

        with Image.open(
            io.BytesIO(result.stdout)
        ) as img:
            return img.convert("RGB").copy()

    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass


# =========================================================
# GORUNTU ON ISLEME
# =========================================================

def uploaded_to_pil(uploaded_file):
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in HEIC_EXTS:
        return decode_heic_bytes(
            data,
            suffix=suffix
        )

    with Image.open(io.BytesIO(data)) as raw:
        return (
            ImageOps
            .exif_transpose(raw)
            .convert("RGB")
            .copy()
        )


def preprocess_pil(img):
    fitted = ImageOps.fit(
        img,
        (SIZE, SIZE),
        method=Image.Resampling.LANCZOS
    )

    arr = (
        np.asarray(
            fitted,
            dtype=np.float32
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        np.transpose(
            arr,
            (2, 0, 1)
        ).copy()
    )

    return tensor.unsqueeze(0)


# =========================================================
# ORTAK ECA
# =========================================================

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = (
            y.squeeze(-1)
            .transpose(-1, -2)
        )

        y = self.conv(y)

        y = (
            y.transpose(-1, -2)
            .unsqueeze(-1)
        )

        y = self.sigmoid(y)

        return x * y.expand_as(x)


# =========================================================
# V3
# =========================================================

class V3ResidualECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.eca = ECAAttention(kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        old = x

        y = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        y = self.bn2(
            self.conv2(y)
        )

        y = self.eca(y)

        return self.relu(
            y + old
        )


class V3Stronger(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block64 = V3ResidualECABlock(64)

        self.down128 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block128 = V3ResidualECABlock(128)

        self.down256 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.block256 = V3ResidualECABlock(256)

        self.avg2 = nn.AdaptiveAvgPool2d((2, 2))
        self.max2 = nn.AdaptiveMaxPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.block64(x)

        x = self.down128(x)
        x = self.block128(x)

        x = self.down256(x)
        x = self.block256(x)

        avg = torch.flatten(
            self.avg2(x),
            1
        )

        mx = torch.flatten(
            self.max2(x),
            1
        )

        x = torch.cat(
            [avg, mx],
            dim=1
        )

        return self.classifier(x).squeeze(1)


# =========================================================
# V4 MULTISCALE512 — CHECKPOINT ILE BIREBIR UYUMLU
# =========================================================

class V4ResidualECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        self.eca = ECAAttention(kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        y = self.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        y = self.bn2(
            self.conv2(y)
        )

        y = self.eca(y)

        return self.relu(
            y + identity
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(attention)
        )

        return x * attention


class LightBottleneck512(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=128,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.expand = nn.Sequential(
            nn.Conv2d(
                128,
                512,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(512)
        )

        self.skip = nn.Sequential(
            nn.Conv2d(
                256,
                512,
                kernel_size=1,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(512)
        )

        self.eca = ECAAttention(kernel_size=3)
        self.spatial = SpatialAttention(kernel_size=7)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.skip(x)

        y = self.reduce(x)
        y = self.depthwise(y)
        y = self.expand(y)

        y = self.eca(y)
        y = self.spatial(y)

        return self.relu(
            y + identity
        )


class V4MultiScale512(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block64 = V4ResidualECABlock(64)

        self.down128 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block128 = V4ResidualECABlock(128)

        self.down256 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.block256 = V4ResidualECABlock(256)

        self.light512 = LightBottleneck512()

        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.max1 = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.LayerNorm(1792),

            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1)
        )

    def pool_both(self, x):
        avg = torch.flatten(
            self.avg1(x),
            1
        )

        mx = torch.flatten(
            self.max1(x),
            1
        )

        return torch.cat(
            [avg, mx],
            dim=1
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block64(x)

        x = self.down128(x)
        f128 = self.block128(x)

        x = self.down256(f128)
        f256 = self.block256(x)

        f512 = self.light512(f256)

        p128 = self.pool_both(f128)
        p256 = self.pool_both(f256)
        p512 = self.pool_both(f512)

        fused = torch.cat(
            [p128, p256, p512],
            dim=1
        )

        return self.classifier(
            fused
        ).squeeze(1)

# =========================================================
# V5
# =========================================================

class V5Extended384(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block64 = V3ResidualECABlock(64)

        self.down128 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.block128 = V3ResidualECABlock(128)

        self.down256 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.block256 = V3ResidualECABlock(256)

        self.down384 = nn.Sequential(
            nn.Conv2d(
                256,
                384,
                3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )

        self.block384 = V3ResidualECABlock(384)

        self.avg2 = nn.AdaptiveAvgPool2d((2, 2))
        self.max2 = nn.AdaptiveMaxPool2d((2, 2))

        self.classifier = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block64(x)

        x = self.down128(x)
        x = self.block128(x)

        x = self.down256(x)
        x = self.block256(x)

        x = self.down384(x)
        x = self.block384(x)

        avg = torch.flatten(
            self.avg2(x),
            1
        )

        mx = torch.flatten(
            self.max2(x),
            1
        )

        x = torch.cat(
            [avg, mx],
            dim=1
        )

        return self.classifier(x).squeeze(1)


MODEL_CLASSES = {
    "V3 Stronger (Final / önerilen)": V3Stronger,
    "V4 MultiScale512": V4MultiScale512,
    "V5 Extended384": V5Extended384,
}


# =========================================================
# CHECKPOINT BULMA
# =========================================================

def get_best_checkpoints(model_dir):
    checkpoints = []

    for fold in range(1, 6):
        fold_dir = model_dir / f"fold_{fold}"

        found = sorted(
            fold_dir.glob(
                "BEST_MODEL_epoch*.pth"
            )
        )

        if len(found) != 1:
            raise RuntimeError(
                f"Fold {fold} icin tek BEST_MODEL bekleniyor. "
                f"Bulunan={len(found)} | {fold_dir}"
            )

        checkpoints.append(found[0])

    return checkpoints


# =========================================================
# TAHMIN
# =========================================================

@torch.no_grad()
def predict_one_model(model, x):
    model.eval()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=(DEVICE.type == "cuda")
    ):
        logits_normal = model(x)

        logits_flip = model(
            torch.flip(
                x,
                dims=[3]
            )
        )

        logits_tta = (
            logits_normal
            +
            logits_flip
        ) / 2.0

    prob_fake = (
        torch.sigmoid(
            logits_tta.float()
        )
        .item()
    )

    return prob_fake


def ensemble_predict(model_name, x):
    model_dir = MODEL_DIRS[model_name]
    model_class = MODEL_CLASSES[model_name]

    checkpoints = get_best_checkpoints(
        model_dir
    )

    fold_probs = []

    for ckpt in checkpoints:
        model = model_class().to(DEVICE)

        try:
            state = torch.load(
                ckpt,
                map_location="cpu",
                weights_only=True
            )
        except TypeError:
            state = torch.load(
                ckpt,
                map_location="cpu"
            )

        model.load_state_dict(state)

        prob = predict_one_model(
            model,
            x
        )

        fold_probs.append(prob)

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return float(
        np.mean(fold_probs)
    ), fold_probs


# =========================================================
# STREAMLIT ARAYUZ
# =========================================================

st.set_page_config(
    page_title="Gerçek / Yapay Görüntü Tespiti",
    page_icon="🧠",
    layout="centered"
)

st.title("Gerçek / Yapay Görüntü Tespiti")
st.caption(
    "CNN tabanlı 5-fold ensemble • TTA aktif • 320×320 giriş"
)

with st.sidebar:
    st.header("Model")

    selected_model = st.selectbox(
        "Kullanılacak model",
        list(MODEL_DIRS.keys()),
        index=0
    )

    st.write(
        f"**Cihaz:** {DEVICE}"
    )

    st.write(
        f"**Karar eşiği:** {THRESHOLD:.2f}"
    )

    st.info(
        "Sunum için öneri: V3 Stronger seçili kalsın. "
        "V4 ve V5 de karşılaştırma için kullanılabilir."
    )

uploaded = st.file_uploader(
    "Bir görüntü yükle",
    type=[
        "jpg",
        "jpeg",
        "png",
        "heic",
        "heif"
    ]
)

if uploaded is None:
    st.info(
        "Test etmek için JPEG, PNG veya HEIC görüntü yükle."
    )

else:
    try:
        pil_img = uploaded_to_pil(uploaded)

        st.image(
            pil_img,
            caption=uploaded.name,
            use_container_width=True
        )

        x = preprocess_pil(
            pil_img
        ).to(DEVICE)

        if st.button(
            "Analiz Et",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                "5 fold model çalıştırılıyor..."
            ):
                prob_fake, fold_probs = ensemble_predict(
                    selected_model,
                    x
                )

            prob_real = 1.0 - prob_fake

            if prob_fake >= THRESHOLD:
                predicted = "FAKE / YAPAY"
                st.error(
                    f"TAHMİN: {predicted}"
                )
            else:
                predicted = "REAL / GERÇEK"
                st.success(
                    f"TAHMİN: {predicted}"
                )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "REAL olasılığı",
                    f"%{prob_real * 100:.2f}"
                )

            with c2:
                st.metric(
                    "FAKE olasılığı",
                    f"%{prob_fake * 100:.2f}"
                )

            st.progress(
                min(
                    max(prob_fake, 0.0),
                    1.0
                ),
                text=(
                    f"FAKE olasılığı: "
                    f"%{prob_fake * 100:.2f}"
                )
            )

            with st.expander(
                "5 fold ayrı sonuçları"
            ):
                for idx, p in enumerate(
                    fold_probs,
                    start=1
                ):
                    label = (
                        "FAKE"
                        if p >= THRESHOLD
                        else "REAL"
                    )

                    st.write(
                        f"Fold {idx}: "
                        f"P(FAKE)=%{p*100:.2f} "
                        f"→ {label}"
                    )

            st.caption(
                "Final karar, 5 fold'un TTA olasılıklarının "
                "ortalaması ile verilir."
            )

    except Exception as e:
        st.exception(e)
