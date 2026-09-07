# -*- coding: utf-8 -*-
"""
V3 + V4 SENTETIK GORUNTU GURULTU (NOISE) DAYANIKLILIK TESTI

NE YAPAR?
- Bagimsiz 24 test setindeki SADECE 12 FAKE/sentetik goruntuyu alir.
- Her goruntunun 3 halini test eder:
    1) Orijinal
    2) Hafif Gaussian noise  (sigma=0.02 ~= 5/255)
    3) Orta Gaussian noise   (sigma=0.05 ~= 13/255)
- V3 Stronger ve V4 MultiScale512'nin daha once egitilmis
  5 fold BEST_MODEL checkpoint'lerini kullanir.
- YENIDEN EGITIM YAPMAZ.
- Her model icin 5-fold ensemble + TTA kullanir.
- FAKE tespit basarisi ve ortalama FAKE olasiligini raporlar.
- Ayni noise matrisi iki modele de uygulanir; karsilastirma adildir.

Beklenen klasor yapisi:

C:\\Users\\Ceren\\Downloads\\yapay sinir aglari\\
|
|-- test\\
|   |-- test_dataset_pairli.csv
|   `-- ...
|
|-- cross_validation_master408\\
|   |-- cnn_v3_stronger_master408\\
|   |   |-- fold_1\\BEST_MODEL_epoch....pth
|   |   ...
|   |
|   `-- cnn_v4_multiscale512\\
|       |-- fold_1\\BEST_MODEL_epoch....pth
|       ...
|
`-- noise_test_V3_V4.py

Cikti:
noise_test_sentetik_V3_V4\\
"""

from pathlib import Path, PureWindowsPath
from contextlib import nullcontext
import io
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# =========================================================
# AYARLAR
# =========================================================

BASE = Path(__file__).resolve().parent

TEST_DIR = BASE / "test"
TEST_CSV = TEST_DIR / "test_dataset_pairli.csv"

V3_DIR = (
    BASE
    / "cross_validation_master408"
    / "cnn_v3_stronger_master408"
)

V4_DIR = (
    BASE
    / "cross_validation_master408"
    / "cnn_v4_multiscale512"
)

V5_DIR = (
    BASE
    / "cross_validation_master408"
    / "cnn_v5_extended384"
)

OUT_DIR = BASE / "noise_test_sentetik_V3_V4_V5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 320
THRESHOLD = 0.50
SEED = 2026

# 0-1 piksel uzayinda:
# 0.02 ~= 5/255
# 0.05 ~= 13/255
NOISE_LEVELS = [
    ("ORIGINAL", 0.00),
    ("HAFIF", 0.02),
    ("ORTA", 0.05),
]

HEIC_EXTS = {".heic", ".heif"}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


# =========================================================
# HEIC / GORUNTU OKUMA
# =========================================================

def find_imagemagick():
    candidates = [
        "magick",
        "magick.exe",
    ]

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

    raise RuntimeError(
        "ImageMagick bulunamadi. HEIC okumak icin 'magick' komutu gerekli."
    )


IMAGEMAGICK = None


def decode_heic_with_imagemagick(path):
    global IMAGEMAGICK

    if IMAGEMAGICK is None:
        IMAGEMAGICK = find_imagemagick()

    # stdout'a PNG yaz
    result = subprocess.run(
        [
            IMAGEMAGICK,
            str(path),
            "png:-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"HEIC acilamadi:\n{path}\n\n"
            + result.stderr.decode(
                "utf-8",
                errors="replace"
            )
        )

    with Image.open(
        io.BytesIO(result.stdout)
    ) as img:
        return img.convert("RGB").copy()


def load_rgb(path):
    path = Path(path)

    if path.suffix.lower() in HEIC_EXTS:
        img = decode_heic_with_imagemagick(path)
        img = ImageOps.fit(
            img,
            (SIZE, SIZE),
            method=Image.Resampling.LANCZOS
        )
    else:
        with Image.open(path) as raw:
            img = (
                ImageOps
                .exif_transpose(raw)
                .convert("RGB")
            )

            img = ImageOps.fit(
                img,
                (SIZE, SIZE),
                method=Image.Resampling.LANCZOS
            )

    arr = (
        np.asarray(
            img,
            dtype=np.float32
        )
        / 255.0
    )

    return torch.from_numpy(
        np.transpose(
            arr,
            (2, 0, 1)
        ).copy()
    )


# =========================================================
# TEST PATH COZUMLEME
# =========================================================

def resolve_test_path(raw_path):
    raw = str(raw_path).strip().strip('"')

    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    relative = TEST_DIR / raw
    if relative.exists():
        return relative.resolve()

    name = PureWindowsPath(raw).name

    matches = [
        p
        for p in TEST_DIR.rglob(name)
        if p.is_file()
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    lower_name = name.casefold()

    matches = [
        p
        for p in TEST_DIR.rglob("*")
        if p.is_file()
        and p.name.casefold() == lower_name
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Test goruntusu bulunamadi:\n{raw}"
    )


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

        return x * y


# =========================================================
# V3 STRONGER — CHECKPOINT ILE BIREBIR UYUMLU
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
# V5 EXTENDED384 — CHECKPOINT ILE BIREBIR UYUMLU
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


# =========================================================
# CHECKPOINT / INFERENCE
# =========================================================

def get_fold_checkpoints(model_dir):
    checkpoints = []

    for fold in range(1, 6):
        fold_dir = model_dir / f"fold_{fold}"

        matches = list(
            fold_dir.glob(
                "BEST_MODEL_epoch*.pth"
            )
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"{model_dir.name} / Fold {fold}: "
                f"tek BEST_MODEL bulunmali, bulunan={len(matches)}"
            )

        checkpoints.append(matches[0])

    return checkpoints


def load_state(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu"
        )


@torch.no_grad()
def predict_tta(model, batch):
    model.eval()

    amp_ctx = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )
        if DEVICE.type == "cuda"
        else nullcontext()
    )

    with amp_ctx:
        logits_normal = model(batch)

        logits_flip = model(
            torch.flip(
                batch,
                dims=[3]
            )
        )

        logits = (
            logits_normal
            +
            logits_flip
        ) / 2.0

    return (
        torch.sigmoid(
            logits.float()
        )
        .detach()
        .cpu()
        .numpy()
    )


def ensemble_predict(
    model_name,
    model_class,
    model_dir,
    all_noise_batch
):
    checkpoints = get_fold_checkpoints(
        model_dir
    )

    fold_probs = []

    print(
        f"\n{model_name} 5-fold ensemble:"
    )

    for fold, ckpt in enumerate(
        checkpoints,
        start=1
    ):
        model = model_class().to(
            DEVICE
        )

        model.load_state_dict(
            load_state(ckpt)
        )

        probs = predict_tta(
            model,
            all_noise_batch
        )

        fold_probs.append(
            probs
        )

        print(
            f"  Fold {fold}: {ckpt.name}"
        )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return np.stack(
        fold_probs,
        axis=0
    ).mean(
        axis=0
    )


# =========================================================
# NOISE URETIMI
# =========================================================

def build_noise_versions(clean_cpu):
    """
    Ayni temel random noise tum seviyelerde kullanilir.
    Boylece HAFIF ve ORTA birbirinin adil, olceklenmis versiyonudur.
    """
    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(SEED)

    base_noise = torch.randn(
        clean_cpu.shape,
        generator=generator,
        dtype=clean_cpu.dtype
    )

    batches = []
    meta = []

    for level_name, sigma in NOISE_LEVELS:
        if sigma == 0:
            noisy = clean_cpu.clone()
        else:
            noisy = torch.clamp(
                clean_cpu
                +
                sigma * base_noise,
                0.0,
                1.0
            )

        batches.append(
            noisy
        )

        for image_idx in range(
            len(clean_cpu)
        ):
            meta.append({
                "noise_level": level_name,
                "sigma": sigma,
                "image_index": image_idx,
            })

    # siralama:
    # 12 original, 12 hafif, 12 orta
    return (
        torch.cat(
            batches,
            dim=0
        ),
        pd.DataFrame(meta)
    )


# =========================================================
# ONIZLEME
# =========================================================

def tensor_to_pil(x):
    arr = (
        x.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    arr = np.clip(
        arr * 255.0,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        arr,
        mode="RGB"
    )


def save_noise_preview(all_versions_cpu):
    """
    Ilk 3 FAKE goruntu icin:
    original | hafif | orta
    """
    n_images = (
        all_versions_cpu.shape[0]
        //
        len(NOISE_LEVELS)
    )

    n_preview = min(
        3,
        n_images
    )

    thumb = 240
    title_h = 35

    canvas = Image.new(
        "RGB",
        (
            thumb * len(NOISE_LEVELS),
            (thumb + title_h) * n_preview
        ),
        "white"
    )

    for row in range(
        n_preview
    ):
        for col, (
            level_name,
            sigma
        ) in enumerate(
            NOISE_LEVELS
        ):
            idx = (
                col * n_images
                +
                row
            )

            img = tensor_to_pil(
                all_versions_cpu[idx]
            )

            img = img.resize(
                (thumb, thumb),
                Image.Resampling.LANCZOS
            )

            canvas.paste(
                img,
                (
                    col * thumb,
                    row * (thumb + title_h)
                    +
                    title_h
                )
            )

            # PIL default font ile sade baslik
            from PIL import ImageDraw
            draw = ImageDraw.Draw(
                canvas
            )

            draw.text(
                (
                    col * thumb + 8,
                    row * (thumb + title_h) + 8
                ),
                f"{level_name} sigma={sigma:.2f}",
                fill="black"
            )

    canvas.save(
        OUT_DIR / "01_noise_ornekleri.png"
    )


# =========================================================
# ANA AKIS
# =========================================================

print("=" * 84)
print("SENTETIK GORUNTU NOISE TESTI — V3 + V4 + V5")
print("=" * 84)
print("Device:", DEVICE)

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Test CSV bulunamadi:\n{TEST_CSV}"
    )

test_df = pd.read_csv(
    TEST_CSV
)

required_cols = {
    "no",
    "file_name",
    "file_path",
    "label",
    "format",
    "generator",
}

missing = required_cols - set(
    test_df.columns
)

if missing:
    raise RuntimeError(
        "Test CSV eksik sutunlar: "
        +
        ", ".join(
            sorted(missing)
        )
    )

test_df["label"] = (
    test_df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)

fake_df = (
    test_df[
        test_df["label"] == "FAKE"
    ]
    .copy()
    .reset_index(drop=True)
)

if len(fake_df) != 12:
    raise RuntimeError(
        f"12 FAKE bekleniyordu, bulunan={len(fake_df)}"
    )

fake_df[
    "resolved_path"
] = [
    str(
        resolve_test_path(p)
    )
    for p
    in fake_df[
        "file_path"
    ]
]

print(
    f"\nSentetik/FAKE test goruntusu: {len(fake_df)}"
)

print(
    "Generator:",
    fake_df[
        "generator"
    ].value_counts().to_dict()
)

print(
    "Format:",
    fake_df[
        "format"
    ].value_counts().to_dict()
)

print(
    "\n12 FAKE goruntu okunuyor..."
)

clean_cpu = torch.stack(
    [
        load_rgb(p)
        for p
        in fake_df[
            "resolved_path"
        ]
    ]
)

all_versions_cpu, meta = (
    build_noise_versions(
        clean_cpu
    )
)

save_noise_preview(
    all_versions_cpu
)

all_versions = (
    all_versions_cpu
    .to(
        DEVICE
    )
)

# =========================================================
# MODELLERI TEST ET
# =========================================================

model_specs = [
    (
        "V3_STRONGER",
        V3Stronger,
        V3_DIR
    ),
    (
        "V4_MULTISCALE512",
        V4MultiScale512,
        V4_DIR
    ),
    (
        "V5_EXTENDED384",
        V5Extended384,
        V5_DIR
    ),
]

all_detail = []
all_summary = []

for (
    model_name,
    model_class,
    model_dir
) in model_specs:

    if not model_dir.exists():
        print(
            f"\nATLANDI: {model_name} klasoru yok:\n{model_dir}"
        )
        continue

    probs = ensemble_predict(
        model_name,
        model_class,
        model_dir,
        all_versions
    )

    detail = meta.copy()

    # Her noise seviyesinde image_index tekrar 0..11
    detail["model"] = model_name
    detail["fake_probability"] = probs
    detail["predicted_label"] = np.where(
        probs >= THRESHOLD,
        "FAKE",
        "REAL"
    )
    detail["correct_fake_detection"] = (
        detail[
            "predicted_label"
        ]
        ==
        "FAKE"
    )

    detail["file_name"] = [
        fake_df.loc[
            i,
            "file_name"
        ]
        for i
        in detail[
            "image_index"
        ]
    ]

    detail["pair_no"] = [
        fake_df.loc[
            i,
            "no"
        ]
        for i
        in detail[
            "image_index"
        ]
    ]

    detail["format"] = [
        fake_df.loc[
            i,
            "format"
        ]
        for i
        in detail[
            "image_index"
        ]
    ]

    detail["generator"] = [
        fake_df.loc[
            i,
            "generator"
        ]
        for i
        in detail[
            "image_index"
        ]
    ]

    all_detail.append(
        detail
    )

    for (
        level_name,
        sigma
    ) in NOISE_LEVELS:
        g = detail[
            detail[
                "noise_level"
            ]
            ==
            level_name
        ]

        correct = int(
            g[
                "correct_fake_detection"
            ].sum()
        )

        all_summary.append({
            "model": model_name,
            "noise_level": level_name,
            "sigma": sigma,
            "fake_correct": correct,
            "fake_total": len(g),
            "fake_detection_accuracy": (
                correct / len(g)
            ),
            "mean_fake_probability": float(
                g[
                    "fake_probability"
                ].mean()
            ),
            "median_fake_probability": float(
                g[
                    "fake_probability"
                ].median()
            ),
            "min_fake_probability": float(
                g[
                    "fake_probability"
                ].min()
            ),
            "max_fake_probability": float(
                g[
                    "fake_probability"
                ].max()
            ),
        })

detail_df = pd.concat(
    all_detail,
    ignore_index=True
)

summary_df = pd.DataFrame(
    all_summary
)

detail_df.to_csv(
    OUT_DIR
    / "02_noise_test_V3_V4_V5_gorsel_bazli.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    OUT_DIR
    / "03_noise_test_V3_V4_V5_ozet.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# GENERATOR + FORMAT ALT GRUPLARI
# =========================================================

subgroup_rows = []

for (
    model_name,
    noise_level,
    generator
), g in detail_df.groupby(
    [
        "model",
        "noise_level",
        "generator"
    ]
):
    subgroup_rows.append({
        "type": "GENERATOR",
        "model": model_name,
        "noise_level": noise_level,
        "group": generator,
        "n": len(g),
        "fake_detection_accuracy": float(
            g[
                "correct_fake_detection"
            ].mean()
        ),
        "mean_fake_probability": float(
            g[
                "fake_probability"
            ].mean()
        ),
    })

for (
    model_name,
    noise_level,
    fmt
), g in detail_df.groupby(
    [
        "model",
        "noise_level",
        "format"
    ]
):
    subgroup_rows.append({
        "type": "FORMAT",
        "model": model_name,
        "noise_level": noise_level,
        "group": fmt,
        "n": len(g),
        "fake_detection_accuracy": float(
            g[
                "correct_fake_detection"
            ].mean()
        ),
        "mean_fake_probability": float(
            g[
                "fake_probability"
            ].mean()
        ),
    })

pd.DataFrame(
    subgroup_rows
).to_csv(
    OUT_DIR
    / "04_noise_test_alt_gruplar.csv",
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# GRAFIK 1 — FAKE TESPIT BASARISI
# =========================================================

levels = [
    x[0]
    for x
    in NOISE_LEVELS
]

models = list(
    summary_df[
        "model"
    ].unique()
)

x = np.arange(
    len(levels)
)

width = (
    0.8
    /
    max(
        len(models),
        1
    )
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

for i, model_name in enumerate(
    models
):
    g = (
        summary_df[
            summary_df[
                "model"
            ]
            ==
            model_name
        ]
        .set_index(
            "noise_level"
        )
        .loc[
            levels
        ]
    )

    vals = (
        100
        *
        g[
            "fake_detection_accuracy"
        ].to_numpy()
    )

    bars = ax.bar(
        x
        +
        (
            i
            -
            (
                len(models) - 1
            )
            / 2
        )
        *
        width,
        vals,
        width=width,
        label=model_name
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3
    )

ax.set_xticks(
    x,
    [
        "Orijinal",
        "Hafif Noise",
        "Orta Noise"
    ]
)

ax.set_ylim(
    0,
    105
)

ax.set_ylabel(
    "FAKE dogru tespit (%)"
)

ax.set_title(
    "Sentetik Goruntulerde Gurultu Dayanikliligi"
)

ax.legend()
ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "05_fake_tespit_vs_noise.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# GRAFIK 2 — ORTALAMA FAKE OLASILIGI
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)

for model_name in models:
    g = (
        summary_df[
            summary_df[
                "model"
            ]
            ==
            model_name
        ]
        .set_index(
            "noise_level"
        )
        .loc[
            levels
        ]
    )

    ax.plot(
        [
            "Orijinal",
            "Hafif Noise",
            "Orta Noise"
        ],
        100
        *
        g[
            "mean_fake_probability"
        ].to_numpy(),
        marker="o",
        linewidth=2,
        label=model_name
    )

ax.axhline(
    50,
    linestyle="--",
    linewidth=1.5,
    label="FAKE threshold %50"
)

ax.set_ylim(
    0,
    100
)

ax.set_ylabel(
    "Ortalama FAKE olasiligi (%)"
)

ax.set_title(
    "Noise Arttikca Modelin FAKE Guveni"
)

ax.legend()
ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "06_ortalama_fake_olasiligi_vs_noise.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# GRAFIK 3 — GORUNTU BAZLI OLASILIK DEGISIMI
# =========================================================

for model_name in models:
    m = detail_df[
        detail_df[
            "model"
        ]
        ==
        model_name
    ]

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    for image_idx in range(
        12
    ):
        g = (
            m[
                m[
                    "image_index"
                ]
                ==
                image_idx
            ]
            .set_index(
                "noise_level"
            )
            .loc[
                levels
            ]
        )

        ax.plot(
            [
                "Orijinal",
                "Hafif Noise",
                "Orta Noise"
            ],
            100
            *
            g[
                "fake_probability"
            ].to_numpy(),
            marker="o",
            alpha=0.75,
            label=str(
                g[
                    "file_name"
                ].iloc[0]
            )
        )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_ylabel(
        "FAKE olasiligi (%)"
    )

    ax.set_title(
        f"{model_name} — 12 Sentetik Goruntunun Noise Altinda Degisimi"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR
        /
        f"07_{model_name}_gorsel_bazli_noise.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =========================================================
# METIN RAPORU
# =========================================================

lines = [
    "SENTETIK GORUNTU GURULTU TESTI — V3 + V4 + V5",
    "",
    "Test seti: Bagimsiz 24 setinin 12 FAKE/sentetik goruntusu",
    "Egitim: YOK",
    "Inference: 5-fold ensemble + horizontal-flip TTA",
    f"Karar threshold: {THRESHOLD:.2f}",
    f"Noise seed: {SEED}",
    "",
    "Noise seviyeleri:",
    "  ORIGINAL: sigma=0.00",
    "  HAFIF   : sigma=0.02 (~5/255)",
    "  ORTA    : sigma=0.05 (~13/255)",
    "",
]

for _, r in summary_df.iterrows():
    lines.append(
        f"{r['model']} | {r['noise_level']} | "
        f"FAKE {int(r['fake_correct'])}/{int(r['fake_total'])} "
        f"(%{100*r['fake_detection_accuracy']:.2f}) | "
        f"Ort. FAKE olasiligi %{100*r['mean_fake_probability']:.2f}"
    )

lines += [
    "",
    "YORUM KURALI:",
    "- Noise arttikca FAKE tespit basarisi ve/veya FAKE olasiligi "
    "azaliyorsa model gurultuye duyarlidir.",
    "- Sonuclar benzer kaliyorsa model bu gurultu seviyelerine daha dayaniklidir.",
    "- Bu test sentetik goruntuleri yeniden egitime katmaz; sadece test eder.",
]

(
    OUT_DIR
    / "08_NOISE_TEST_RAPOR.txt"
).write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)

print("\n" + "=" * 84)
print("NOISE TESTI TAMAMLANDI")
print("=" * 84)

for _, r in summary_df.iterrows():
    print(
        f"{r['model']:18s} | "
        f"{r['noise_level']:8s} | "
        f"FAKE {int(r['fake_correct'])}/"
        f"{int(r['fake_total'])} "
        f"(%{100*r['fake_detection_accuracy']:.2f}) | "
        f"Ort.P(fake) %{100*r['mean_fake_probability']:.2f}"
    )

print("\nCikti klasoru:")
print(OUT_DIR)
