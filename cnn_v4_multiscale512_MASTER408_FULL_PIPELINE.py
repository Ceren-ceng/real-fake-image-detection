# -*- coding: utf-8 -*-
"""
CNN-v4 MULTISCALE512 — RESIDUAL+ECA+SPATIAL — MASTER 408 / GROUP-AWARE 5-FOLD / FULL PIPELINE

Bu sürüm, önceki cnn_v3_stronger_PAIRLI_5fold_pytorch.py mimarisini ve
temel eğitim ayarlarını KORUR.

Yeni veri düzeni:
- master_dataset_408.csv
- 408 görüntü = 204 REAL + 204 FAKE
- 240 PAIRED + 168 UNPAIRED
- group_id:
    pair_XXX       -> aynı eşli REAL+FAKE aynı fold'da kalır
    unpaired_XXX   -> tek görüntülük bağımsız grup

Yeni eklenenler:
1) StratifiedGroupKFold ile 5-fold otomatik üretilir.
2) Her görüntü için OOF (out-of-fold) tahmini kaydedilir.
3) TTA ve NO-TTA sonuçları ayrı kaydedilir.
4) PAIRED/UNPAIRED, HEIC/JPEG/PNG ve generator bazlı rapor çıkar.
5) human_test_mapping_24.csv varsa insan testindeki 24 görüntünün
   OOF sonuçlarını otomatik ayırır.
6) Taşınmış klasörler için eski file_path yollarını BASE klasörüne
   göre otomatik düzeltmeye çalışır.
7) Windows pillow-heif DLL engeline takılmamak için HEIC dosyalarını
   ImageMagick ile decode eder. Model/preprocessing mantığı değişmez.

NOT:
- Bu dosya bağımsız 24 final test setini KULLANMAZ.
- Final test daha sonra ayrı yapılacaktır.
"""

from pathlib import Path, PureWindowsPath
import io
import os
import random
import shutil
import copy
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# AYARLAR
# =========================================================

BASE = Path(__file__).resolve().parent

MASTER_CSV = BASE / "master_dataset_408.csv"
HUMAN_MAPPING_CSV = BASE / "human_test_mapping_24.csv"

OUT_DIR = BASE / "cross_validation_master408" / "cnn_v4_multiscale512"

SIZE = 320
BATCH = 8

MAX_EPOCHS = 200
PATIENCE = 60

LR = 0.0001
MIN_LR = 0.000001
WEIGHT_DECAY = 0.0003

LABEL_SMOOTH = 0.00
HFLIP_PROB = 0.50
SHIFT_PIXELS = 8

SEED = 42
PRINT_EVERY = 5

LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1
}

LABEL_NAME = {
    0: "REAL",
    1: "FAKE"
}

# Taşınmış klasörlerde eski absolute path'i BASE altına rebasing için.
KNOWN_DATA_ROOTS = [
    "görseller",
    "real_fake_esli",
    "esiz_dataset_168_duzenlenmis",
]

HEIC_EXTS = {".heic", ".heif"}


# =========================================================
# SABİTLİK / GPU
# =========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU bulunamadı.")

DEVICE = torch.device("cuda")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================================================
# PATH ÇÖZÜMLEME
# =========================================================

def resolve_dataset_path(raw_path):
    """
    CSV içindeki eski Windows absolute path'i önce olduğu gibi dener.
    Bulunmazsa görseller / real_fake_esli / esiz_dataset_168_duzenlenmis
    marker'larından itibaren BASE altına taşır.
    """
    raw = str(raw_path).strip()

    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    wp = PureWindowsPath(raw)
    parts = list(wp.parts)
    folded = [str(x).casefold() for x in parts]

    for marker in KNOWN_DATA_ROOTS:
        m = marker.casefold()

        for i, part in enumerate(folded):
            if part == m:
                candidate = BASE.joinpath(*parts[i:])
                if candidate.exists():
                    return candidate.resolve()

    # Son çare: dosya adını BASE altında ara.
    # Tek eşleşme varsa kullan.
    name = wp.name
    if name:
        matches = [
            p for p in BASE.rglob(name)
            if p.is_file()
        ]

        # Çıktı klasörlerinin içindeki olası kopyaları dışla.
        matches = [
            p for p in matches
            if "cross_validation_master408" not in str(p)
        ]

        if len(matches) == 1:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Dosya bulunamadı ve otomatik çözümlenemedi:\n{raw}"
    )


def prepare_master_dataframe():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"master_dataset_408.csv bulunamadı:\n{MASTER_CSV}"
        )

    df = pd.read_csv(MASTER_CSV)

    required = {
        "id",
        "file_name",
        "file_path",
        "label",
        "format",
        "generator",
        "pair_status",
        "group_id",
    }

    missing_cols = required - set(df.columns)
    if missing_cols:
        raise RuntimeError(
            "Master CSV eksik sütunlar: "
            + ", ".join(sorted(missing_cols))
        )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["format"] = (
        df["format"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["pair_status"] = (
        df["pair_status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["group_id"] = df["group_id"].astype(str)

    df["label_num"] = df["label"].map(LABEL_MAP)

    if df["label_num"].isna().any():
        bad = df.loc[df["label_num"].isna(), "label"].unique()
        raise RuntimeError(f"Bilinmeyen label: {bad}")

    if len(df) != 408:
        raise RuntimeError(
            f"Master CSV 408 satır olmalı. Bulunan: {len(df)}"
        )

    if (df["label"] == "REAL").sum() != 204:
        raise RuntimeError("REAL sayısı 204 değil.")

    if (df["label"] == "FAKE").sum() != 204:
        raise RuntimeError("FAKE sayısı 204 değil.")

    print("\nDosya yolları kontrol ediliyor...")

    resolved = []

    for i, p in enumerate(df["file_path"], start=1):
        rp = resolve_dataset_path(p)
        resolved.append(str(rp))

        if i % 50 == 0 or i == len(df):
            print(f"  path: {i}/{len(df)}")

    df["resolved_file_path"] = resolved

    # Güncellenmiş yolların kopyasını kaydet.
    resolved_csv = BASE / "master_dataset_408_RESOLVED.csv"
    df.drop(columns=["label_num"]).to_csv(
        resolved_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Çözümlenmiş master CSV: {resolved_csv}")

    return df


# =========================================================
# GÖRÜNTÜ OKUMA
# =========================================================

def decode_heic_with_imagemagick(path):
    """
    pillow-heif kullanılmaz.
    ImageMagick HEIC'i PNG byte stream'e decode eder.
    """
    if shutil.which("magick") is None:
        raise RuntimeError(
            "HEIC okumak için ImageMagick gerekli. "
            "'magick -version' çalışmalı."
        )

    result = subprocess.run(
        [
            "magick",
            str(path),
            "-auto-orient",
            "png:-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"HEIC açılamadı:\n{path}\n\n"
            + result.stderr.decode(
                "utf-8",
                errors="replace"
            )
        )

    with Image.open(io.BytesIO(result.stdout)) as img:
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
# ECA
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
# RESIDUAL + ECA BLOK
# =========================================================

class ResidualECABlock(nn.Module):
    """
    V3'te başarılı olan residual + ECA yapısı aynen korunur.
    """
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
    """
    Hafif uzamsal dikkat (spatial attention).

    ECA:
        "Hangi kanal önemli?"
    Spatial Attention:
        "Görüntünün hangi bölgesi önemli?"

    Sadece en derin 512 seviyesinde kullanılır; bu yüzden maliyeti düşüktür.
    """
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
    """
    V4'ün yeni 512-kanal bloğu.

    FULL 512x512 3x3 convolution KULLANMIYORUZ.
    Bu yüzden önceki ağır V4 gibi aşırı yavaşlamaması hedeflenir.

    Ana yol:
        256
         ↓
        1x1 Conv: 256 -> 128
         ↓
        Depthwise 3x3 stride=2: 128 -> 128
         ↓
        1x1 Conv: 128 -> 512
         ↓
        ECA
         ↓
        Spatial Attention

    Skip:
        256 -> 512, 1x1 stride=2

    Çıkış:
        512 x 40 x 40
    """
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


# =========================================================
# CNN-v4 MULTISCALE512
#
# V3'TEN KORUNAN:
# - 320x320 RGB giriş
# - BatchNorm
# - Residual + ECA: 64 -> 128 -> 256
# - AdamW, LR=1e-4, weight decay=3e-4
# - HFlip + ±8 px shift
# - Cosine LR
# - Max 200 epoch / patience 60
# - Group-aware 5-fold
#
# V4'TE YENİ:
# 1) Hafif 256 -> 512 bottleneck residual stage
# 2) 512 seviyesinde ECA + Spatial Attention
# 3) Multi-scale fusion:
#       128 seviye + 256 seviye + 512 seviye
#    Her seviyeden Global AvgPool + Global MaxPool alınır.
# 4) Label smoothing kapalı: 0.00
#
# Fusion boyutu:
# 128*2 + 256*2 + 512*2 = 1792
# =========================================================

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

        self.block64 = ResidualECABlock(64)

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

        self.block128 = ResidualECABlock(128)

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

        self.block256 = ResidualECABlock(256)

        # V4: yeni, hafif 512 seviyeli residual bottleneck
        self.light512 = LightBottleneck512()

        # Aynı global pooling modülleri her ölçekte kullanılabilir.
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.max1 = nn.AdaptiveMaxPool2d((1, 1))

        # Multi-scale:
        # 128 avg/max = 256
        # 256 avg/max = 512
        # 512 avg/max = 1024
        # toplam = 1792
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
# BATCH / METRİK YARDIMCILARI
# =========================================================

def make_batches(indices, shuffle):
    if shuffle:
        indices = indices[
            torch.randperm(
                len(indices),
                device=DEVICE
            )
        ]

    for start in range(
        0,
        len(indices),
        BATCH
    ):
        yield indices[
            start:
            start + BATCH
        ]


def class_counts(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)

    real_total = int((true == 0).sum())
    fake_total = int((true == 1).sum())

    real_correct = int(
        (
            (true == 0)
            &
            (pred == 0)
        ).sum()
    )

    fake_correct = int(
        (
            (true == 1)
            &
            (pred == 1)
        ).sum()
    )

    return {
        "real_correct": real_correct,
        "real_total": real_total,
        "real_wrong": real_total - real_correct,

        "fake_correct": fake_correct,
        "fake_total": fake_total,
        "fake_wrong": fake_total - fake_correct
    }


# =========================================================
# AUGMENTATION
# =========================================================

def apply_train_augmentation(x):
    if SHIFT_PIXELS > 0:
        pad = SHIFT_PIXELS

        padded = F.pad(
            x,
            (pad, pad, pad, pad),
            mode="reflect"
        )

        offsets = torch.randint(
            low=0,
            high=2 * pad + 1,
            size=(x.shape[0], 2),
            device=x.device
        )

        crops = []

        for i in range(x.shape[0]):
            oy = int(offsets[i, 0].item())
            ox = int(offsets[i, 1].item())

            crops.append(
                padded[
                    i,
                    :,
                    oy:oy + SIZE,
                    ox:ox + SIZE
                ]
            )

        x = torch.stack(crops, dim=0)

    if HFLIP_PROB > 0:
        flip_mask = (
            torch.rand(
                x.shape[0],
                device=x.device
            )
            <
            HFLIP_PROB
        )

        if flip_mask.any():
            x = x.clone()

            x[flip_mask] = torch.flip(
                x[flip_mask],
                dims=[3]
            )

    return x


def smooth_targets(y):
    return (
        y
        * (1.0 - 2.0 * LABEL_SMOOTH)
        +
        LABEL_SMOOTH
    )


# =========================================================
# TRAIN
# =========================================================

def train_epoch(
    model,
    indices,
    criterion,
    optimizer,
    scaler
):
    model.train()

    total_loss = 0.0
    total = 0

    true_all = []
    pred_all = []

    for idx in make_batches(
        indices,
        shuffle=True
    ):
        x = X[idx]
        y = Y[idx]

        x = apply_train_augmentation(x)

        y_smooth = smooth_targets(y)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(x)

            loss = criterion(
                logits,
                y_smooth
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        scaler.step(optimizer)
        scaler.update()

        pred = (
            torch.sigmoid(
                logits.detach().float()
            )
            >= 0.5
        ).long()

        total_loss += loss.item() * len(idx)
        total += len(idx)

        true_all.extend(
            y.detach()
             .cpu()
             .numpy()
             .astype(int)
             .tolist()
        )

        pred_all.extend(
            pred.detach()
                .cpu()
                .numpy()
                .tolist()
        )

    return {
        "loss": total_loss / total,

        "accuracy": accuracy_score(
            true_all,
            pred_all
        ),

        "counts": class_counts(
            true_all,
            pred_all
        )
    }


# =========================================================
# VALIDATION
# =========================================================

def evaluate(
    model,
    indices,
    criterion,
    tta=True
):
    model.eval()

    total_loss = 0.0
    total = 0

    true_all = []
    pred_all = []
    prob_all = []

    with torch.no_grad():
        for idx in make_batches(
            indices,
            shuffle=False
        ):
            x = X[idx]
            y = Y[idx]

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits_normal = model(x)

                if tta:
                    x_flip = torch.flip(
                        x,
                        dims=[3]
                    )

                    logits_flip = model(x_flip)

                    logits = (
                        logits_normal
                        +
                        logits_flip
                    ) / 2.0

                else:
                    logits = logits_normal

                loss = criterion(
                    logits,
                    y
                )

            prob = torch.sigmoid(
                logits.float()
            )

            pred = (
                prob >= 0.5
            ).long()

            total_loss += loss.item() * len(idx)
            total += len(idx)

            true_all.extend(
                y.cpu()
                 .numpy()
                 .astype(int)
                 .tolist()
            )

            pred_all.extend(
                pred.cpu()
                    .numpy()
                    .tolist()
            )

            prob_all.extend(
                prob.cpu()
                    .numpy()
                    .tolist()
            )

    macro_f1 = f1_score(
        true_all,
        pred_all,
        average="macro",
        zero_division=0
    )

    auc = roc_auc_score(
        true_all,
        prob_all
    )

    return {
        "loss": total_loss / total,

        "accuracy": accuracy_score(
            true_all,
            pred_all
        ),

        "precision": precision_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "recall": recall_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "f1": f1_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "macro_f1": macro_f1,
        "auc": auc,

        "true": true_all,
        "pred": pred_all,
        "prob": prob_all,

        "counts": class_counts(
            true_all,
            pred_all
        )
    }


def print_epoch(
    epoch,
    tr,
    va,
    wait,
    best_acc,
    current_lr
):
    tc = tr["counts"]
    vc = va["counts"]

    train_correct = (
        tc["real_correct"]
        +
        tc["fake_correct"]
    )

    val_correct = (
        vc["real_correct"]
        +
        vc["fake_correct"]
    )

    train_total = (
        tc["real_total"]
        +
        tc["fake_total"]
    )

    val_total = (
        vc["real_total"]
        +
        vc["fake_total"]
    )

    print(
        f"\nEpoch {epoch:03d}/{MAX_EPOCHS}"
    )

    print(
        f"  TRAIN | "
        f"Doğru {train_correct}/{train_total} "
        f"(%{tr['accuracy']*100:.1f}) | "
        f"REAL {tc['real_correct']}/{tc['real_total']} | "
        f"FAKE {tc['fake_correct']}/{tc['fake_total']} | "
        f"Loss {tr['loss']:.4f}"
    )

    print(
        f"  VAL   | "
        f"Doğru {val_correct}/{val_total} "
        f"(%{va['accuracy']*100:.1f}) | "
        f"REAL {vc['real_correct']}/{vc['real_total']} | "
        f"FAKE {vc['fake_correct']}/{vc['fake_total']} | "
        f"F1macro {va['macro_f1']:.3f} | "
        f"Loss {va['loss']:.4f} | "
        f"Best %{best_acc*100:.1f} | "
        f"LR {current_lr:.7f} | "
        f"Bekleme {wait}/{PATIENCE}"
    )


# =========================================================
# GRAFİK
# =========================================================

def save_fold_plot(
    history,
    final,
    fold,
    folder,
    best_epoch,
    stopped_epoch
):
    cm = confusion_matrix(
        final["true"],
        final["pred"],
        labels=[0, 1]
    )

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    ax[0, 0].plot(
        epochs,
        history["train_acc"],
        label="Train"
    )

    ax[0, 0].plot(
        epochs,
        history["val_acc"],
        label="Validation"
    )

    ax[0, 0].axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch: {best_epoch}"
    )

    ax[0, 0].set_title(
        f"Fold {fold} - Doğruluk"
    )

    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    ax[0, 1].plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )

    ax[0, 1].axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch: {best_epoch}"
    )

    ax[0, 1].set_title(
        f"Fold {fold} - Loss"
    )

    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    ax[1, 0].imshow(cm)

    ax[1, 0].set_title(
        "Karışıklık Matrisi"
    )

    ax[1, 0].set_xticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    ax[1, 0].set_yticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    ax[1, 0].set_xlabel("Tahmin")
    ax[1, 0].set_ylabel("Gerçek")

    for i in range(2):
        for j in range(2):
            ax[1, 0].text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=15
            )

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Macro-F1",
        "ROC-AUC"
    ]

    vals = [
        final["accuracy"],
        final["precision"],
        final["recall"],
        final["f1"],
        final["macro_f1"],
        final["auc"]
    ]

    bars = ax[1, 1].bar(
        names,
        vals
    )

    ax[1, 1].set_ylim(
        0,
        1.05
    )

    ax[1, 1].set_title(
        "Seçilen Model Metrikleri"
    )

    ax[1, 1].tick_params(
        axis="x",
        rotation=25
    )

    for bar, value in zip(
        bars,
        vals
    ):
        ax[1, 1].text(
            bar.get_x()
            + bar.get_width()/2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=8
        )

    fig.suptitle(
        "CNN-v4 MultiScale512 | "
        f"Master408 Group-Aware Fold {fold} | "
        f"Best={best_epoch}, Stop={stopped_epoch}",
        fontsize=14
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.savefig(
        folder
        / f"fold_{fold}_CNN_V4_GN_DEEP384_OZET.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# =========================================================
# OOF ÖZET
# =========================================================

def summarize_prediction_df(prediction_df):
    true_all = [
        LABEL_MAP[x]
        for x
        in prediction_df["true_label"]
    ]

    pred_all = [
        LABEL_MAP[x]
        for x
        in prediction_df["predicted_label"]
    ]

    prob_all = (
        prediction_df["fake_probability"]
        .astype(float)
        .tolist()
    )

    general = {
        "accuracy": accuracy_score(
            true_all,
            pred_all
        ),

        "precision": precision_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "recall": recall_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "f1": f1_score(
            true_all,
            pred_all,
            zero_division=0
        ),

        "macro_f1": f1_score(
            true_all,
            pred_all,
            average="macro",
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            true_all,
            prob_all
        )
    }

    counts = class_counts(
        true_all,
        pred_all
    )

    cm = confusion_matrix(
        true_all,
        pred_all,
        labels=[0, 1]
    )

    return general, counts, cm


def subgroup_table(pred_df):
    """
    PAIRED/UNPAIRED, format ve generator bazında
    NO-TTA OOF doğruluğu.
    """
    rows = []

    def add_group(group_type, group_value, sub):
        n = len(sub)

        if n == 0:
            return

        correct = (
            sub["true_label"]
            ==
            sub["predicted_label"]
        ).sum()

        real = sub[
            sub["true_label"] == "REAL"
        ]

        fake = sub[
            sub["true_label"] == "FAKE"
        ]

        real_acc = (
            (
                real["true_label"]
                ==
                real["predicted_label"]
            ).mean()
            if len(real)
            else np.nan
        )

        fake_acc = (
            (
                fake["true_label"]
                ==
                fake["predicted_label"]
            ).mean()
            if len(fake)
            else np.nan
        )

        rows.append({
            "group_type": group_type,
            "group_value": group_value,
            "n": n,
            "accuracy": correct / n,
            "real_n": len(real),
            "real_accuracy": real_acc,
            "fake_n": len(fake),
            "fake_accuracy": fake_acc,
        })

    add_group(
        "ALL",
        "ALL",
        pred_df
    )

    for value, sub in pred_df.groupby("pair_status"):
        add_group(
            "pair_status",
            value,
            sub
        )

    for value, sub in pred_df.groupby("format"):
        add_group(
            "format",
            value,
            sub
        )

    # Generator için REAL/NONE ve FAKE üreticiler ayrı görülsün.
    for value, sub in pred_df.groupby("generator"):
        add_group(
            "generator",
            value,
            sub
        )

    return pd.DataFrame(rows)


# =========================================================
# ANA VERİYİ HAZIRLA
# =========================================================

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = prepare_master_dataframe()

print(
    "\nGPU:",
    torch.cuda.get_device_name(0)
)

print(
    "\n408 görüntü 1 kez okunuyor "
    "ve GPU'ya yükleniyor..."
)

images = []
labels = []

for i, row in df.iterrows():
    images.append(
        load_rgb(
            row["resolved_file_path"]
        )
    )

    labels.append(
        int(row["label_num"])
    )

    if (
        (i + 1) % 20 == 0
        or
        i + 1 == len(df)
    ):
        print(
            f"{i+1}/{len(df)}"
        )

X = torch.stack(images).to(DEVICE)

Y = torch.tensor(
    labels,
    dtype=torch.float32,
    device=DEVICE
)

del images
del labels

print("\nHazır.")

print(
    "RGB:",
    tuple(X.shape),
    "|",
    X.device
)

print(
    "REAL:",
    int((Y == 0).sum()),
    "| FAKE:",
    int((Y == 1).sum())
)

print(
    f"Size={SIZE} | "
    f"Batch={BATCH} | "
    f"LR={LR} | "
    f"Weight decay={WEIGHT_DECAY} | "
    f"Label smoothing={LABEL_SMOOTH} | "
    f"Shift=±{SHIFT_PIXELS}px | "
    f"Max epoch={MAX_EPOCHS} | "
    f"Early stopping={PATIENCE} | "
    f"TTA ve NO-TTA ayrı raporlanacak | "
    f"AMP=açık"
)


# =========================================================
# PARAMETRE SAYISI
# =========================================================

temp_model = V4MultiScale512()

param_count = sum(
    p.numel()
    for p in temp_model.parameters()
    if p.requires_grad
)

del temp_model

print(
    "\nCNN-v4 MultiScale512 "
    "öğrenilebilir parametre sayısı: "
    f"{param_count:,}"
)


# =========================================================
# GROUP-AWARE 5 FOLD OLUŞTUR
# =========================================================

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

splits = list(
    splitter.split(
        X=np.zeros(len(df)),
        y=df["label_num"].values,
        groups=df["group_id"].values
    )
)

fold_assignment = np.zeros(
    len(df),
    dtype=int
)

for fold, (_, val_np) in enumerate(
    splits,
    start=1
):
    fold_assignment[val_np] = fold

df["oof_fold"] = fold_assignment

# Aynı group iki fold'a kaçmış mı?
group_fold_counts = (
    df.groupby("group_id")["oof_fold"]
    .nunique()
)

if (group_fold_counts > 1).any():
    bad = group_fold_counts[
        group_fold_counts > 1
    ]
    raise RuntimeError(
        "Group leakage bulundu:\n"
        + bad.to_string()
    )

df.drop(columns=["label_num"]).to_csv(
    OUT_DIR / "fold_assignments_408.csv",
    index=False,
    encoding="utf-8-sig"
)

# V3 ile TAM AYNI foldları kullandığımızı kontrol et.
v3_fold_file = (
    BASE
    / "cross_validation_master408"
    / "cnn_v3_stronger_master408"
    / "fold_assignments_408.csv"
)

if v3_fold_file.exists():
    try:
        old_folds = pd.read_csv(v3_fold_file)

        cmp = df[["id", "oof_fold"]].merge(
            old_folds[["id", "oof_fold"]],
            on="id",
            how="inner",
            suffixes=("_v4", "_v3")
        )

        same_folds = (
            len(cmp) == 408
            and
            (cmp["oof_fold_v4"] == cmp["oof_fold_v3"]).all()
        )

        print(
            "\nV3 ile fold karşılaştırması:",
            "AYNI ✓" if same_folds else "FARKLI ⚠"
        )

        if not same_folds:
            print(
                "UYARI: V3 ve V4 fold atamaları birebir aynı değil. "
                "Karşılaştırma yine yapılabilir ancak birebir fold eşliği yok."
            )
    except Exception as e:
        print(
            "\nV3 fold karşılaştırması yapılamadı:",
            e
        )

print(
    "\n5-fold dağılımı oluşturuldu "
    "(aynı pair aynı fold'da):"
)

for fold in range(1, 6):
    sub = df[df["oof_fold"] == fold]

    print(
        f"  Fold {fold}: "
        f"{len(sub)} görüntü | "
        f"REAL={(sub['label']=='REAL').sum()} | "
        f"FAKE={(sub['label']=='FAKE').sum()} | "
        f"PAIRED={(sub['pair_status']=='PAIRED').sum()} | "
        f"UNPAIRED={(sub['pair_status']=='UNPAIRED').sum()} | "
        f"HEIC={(sub['format']=='HEIC').sum()} | "
        f"JPEG={(sub['format']=='JPEG').sum()} | "
        f"PNG={(sub['format']=='PNG').sum()}"
    )


# =========================================================
# 5 FOLD EĞİTİM
# =========================================================

fold_results = []
all_predictions_tta = []
all_predictions_no_tta = []


for fold, (train_np, val_np) in enumerate(
    splits,
    start=1
):
    print(
        "\n"
        +
        "=" * 86
    )

    print(
        f"CNN-v4 MULTISCALE512 | "
        f"MASTER408 GROUP-AWARE | "
        f"FOLD {fold}/5"
    )

    print(
        "=" * 86
    )

    fold_dir = OUT_DIR / f"fold_{fold}"

    fold_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df = df.iloc[train_np].copy()
    val_df = df.iloc[val_np].copy()

    train_df.drop(columns=["label_num"]).to_csv(
        fold_dir / "train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    val_df.drop(columns=["label_num"]).to_csv(
        fold_dir / "validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    train_idx = torch.tensor(
        train_np,
        dtype=torch.long,
        device=DEVICE
    )

    val_idx = torch.tensor(
        val_np,
        dtype=torch.long,
        device=DEVICE
    )

    torch.manual_seed(SEED + fold)
    torch.cuda.manual_seed_all(SEED + fold)

    model = V4MultiScale512().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=MIN_LR
        )
    )

    scaler = torch.amp.GradScaler("cuda")

    print(
        f"Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)}"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }

    best_acc = -1.0
    best_macro_f1 = -1.0
    best_val_loss = float("inf")

    best_epoch = 0
    best_state = None

    patience_counter = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):
        tr = train_epoch(
            model,
            train_idx,
            criterion,
            optimizer,
            scaler
        )

        # V3 ile aynı checkpoint seçim mantığı korunuyor.
        va = evaluate(
            model,
            val_idx,
            criterion,
            tta=True
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history["train_loss"].append(
            tr["loss"]
        )

        history["val_loss"].append(
            va["loss"]
        )

        history["train_acc"].append(
            tr["accuracy"]
        )

        history["val_acc"].append(
            va["accuracy"]
        )

        history["val_macro_f1"].append(
            va["macro_f1"]
        )

        history["lr"].append(
            current_lr
        )

        accuracy_better = (
            va["accuracy"]
            >
            best_acc + 1e-12
        )

        same_acc_better_macro = (
            abs(
                va["accuracy"] - best_acc
            )
            < 1e-12
            and
            va["macro_f1"]
            >
            best_macro_f1 + 1e-12
        )

        same_acc_same_macro_better_loss = (
            abs(
                va["accuracy"] - best_acc
            )
            < 1e-12
            and
            abs(
                va["macro_f1"] - best_macro_f1
            )
            < 1e-12
            and
            va["loss"]
            <
            best_val_loss - 0.0001
        )

        if (
            accuracy_better
            or
            same_acc_better_macro
            or
            same_acc_same_macro_better_loss
        ):
            best_acc = va["accuracy"]
            best_macro_f1 = va["macro_f1"]
            best_val_loss = va["loss"]
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:
            patience_counter += 1

        scheduler.step()

        if (
            epoch == 1
            or
            epoch % PRINT_EVERY == 0
        ):
            print_epoch(
                epoch,
                tr,
                va,
                patience_counter,
                best_acc,
                current_lr
            )

        if patience_counter >= PATIENCE:
            stopped_epoch = epoch

            print(
                "\nEARLY STOPPING: "
                f"{PATIENCE} epoch iyileşme yok."
            )

            print(
                f"Seçilen epoch: {best_epoch} | "
                f"VAL accuracy %{best_acc*100:.1f} | "
                f"macro-F1={best_macro_f1:.3f}"
            )

            break

    model.load_state_dict(best_state)

    final_tta = evaluate(
        model,
        val_idx,
        criterion,
        tta=True
    )

    final_no_tta = evaluate(
        model,
        val_idx,
        criterion,
        tta=False
    )

    torch.save(
        best_state,
        fold_dir
        / f"BEST_MODEL_epoch{best_epoch}.pth"
    )

    pd.DataFrame({
        "epoch": range(
            1,
            len(history["train_loss"]) + 1
        ),
        "train_loss": history["train_loss"],
        "validation_loss": history["val_loss"],
        "train_accuracy": history["train_acc"],
        "validation_accuracy": history["val_acc"],
        "validation_macro_f1": history["val_macro_f1"],
        "learning_rate": history["lr"]
    }).to_csv(
        fold_dir / "epoch_gecmisi.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------
    # TTA OOF
    # -------------------------
    pred_tta = (
        val_df
        .drop(columns=["label_num"])
        .copy()
    )

    pred_tta["fold"] = fold
    pred_tta["best_epoch"] = best_epoch

    pred_tta["true_label"] = [
        LABEL_NAME[x]
        for x in final_tta["true"]
    ]

    pred_tta["predicted_label"] = [
        LABEL_NAME[x]
        for x in final_tta["pred"]
    ]

    pred_tta["fake_probability"] = (
        final_tta["prob"]
    )

    pred_tta["correct"] = (
        pred_tta["true_label"]
        ==
        pred_tta["predicted_label"]
    )

    pred_tta.to_csv(
        fold_dir / "validation_OOF_TTA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_predictions_tta.append(
        pred_tta
    )

    # -------------------------
    # NO-TTA OOF
    # -------------------------
    pred_no_tta = (
        val_df
        .drop(columns=["label_num"])
        .copy()
    )

    pred_no_tta["fold"] = fold
    pred_no_tta["best_epoch"] = best_epoch

    pred_no_tta["true_label"] = [
        LABEL_NAME[x]
        for x in final_no_tta["true"]
    ]

    pred_no_tta["predicted_label"] = [
        LABEL_NAME[x]
        for x in final_no_tta["pred"]
    ]

    pred_no_tta["fake_probability"] = (
        final_no_tta["prob"]
    )

    pred_no_tta["correct"] = (
        pred_no_tta["true_label"]
        ==
        pred_no_tta["predicted_label"]
    )

    pred_no_tta.to_csv(
        fold_dir / "validation_OOF_NO_TTA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_predictions_no_tta.append(
        pred_no_tta
    )

    c = final_tta["counts"]

    fold_results.append({
        "fold": fold,
        "train_count": len(train_idx),
        "validation_count": len(val_idx),
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,

        "accuracy_TTA": final_tta["accuracy"],
        "accuracy_NO_TTA": final_no_tta["accuracy"],

        "precision_TTA": final_tta["precision"],
        "recall_TTA": final_tta["recall"],
        "f1_TTA": final_tta["f1"],
        "macro_f1_TTA": final_tta["macro_f1"],
        "roc_auc_TTA": final_tta["auc"],

        "real_correct_TTA": c["real_correct"],
        "real_total_TTA": c["real_total"],
        "fake_correct_TTA": c["fake_correct"],
        "fake_total_TTA": c["fake_total"],
    })

    save_fold_plot(
        history,
        final_tta,
        fold,
        fold_dir,
        best_epoch,
        stopped_epoch
    )

    print(
        f"\nFOLD {fold}: "
        f"TTA=%{final_tta['accuracy']*100:.1f} | "
        f"NO-TTA=%{final_no_tta['accuracy']*100:.1f} | "
        f"Best epoch={best_epoch}"
    )

    del model
    torch.cuda.empty_cache()


# =========================================================
# GENEL OOF SONUÇLARI
# =========================================================

results = pd.DataFrame(
    fold_results
)

results.to_csv(
    OUT_DIR / "5_fold_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

oof_tta = pd.concat(
    all_predictions_tta,
    ignore_index=True
).sort_values("id")

oof_no_tta = pd.concat(
    all_predictions_no_tta,
    ignore_index=True
).sort_values("id")

if len(oof_tta) != 408:
    raise RuntimeError(
        f"OOF TTA satırı 408 değil: {len(oof_tta)}"
    )

if len(oof_no_tta) != 408:
    raise RuntimeError(
        f"OOF NO-TTA satırı 408 değil: {len(oof_no_tta)}"
    )

if oof_tta["id"].nunique() != 408:
    raise RuntimeError(
        "OOF TTA'da duplicate/missing id var."
    )

if oof_no_tta["id"].nunique() != 408:
    raise RuntimeError(
        "OOF NO-TTA'da duplicate/missing id var."
    )

oof_tta.to_csv(
    OUT_DIR / "OOF_ALL_408_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)

oof_no_tta.to_csv(
    OUT_DIR / "OOF_ALL_408_NO_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)

general_tta, gc_tta, cm_tta = (
    summarize_prediction_df(
        oof_tta
    )
)

general_no_tta, gc_no_tta, cm_no_tta = (
    summarize_prediction_df(
        oof_no_tta
    )
)



# =========================================================
# OOF'TAN OPTİMAL THRESHOLD BUL
# Bağımsız 24 test setine bakmadan, yalnızca 408 OOF kullanılır.
# Ana amaç: Macro-F1'i maksimize etmek.
# Eşitlikte accuracy, sonra 0.50'ye yakınlık kullanılır.
# =========================================================

def find_optimal_threshold_from_oof(pred_df):
    y_true = np.array(
        [
            LABEL_MAP[x]
            for x in pred_df["true_label"]
        ],
        dtype=int
    )

    probs = (
        pred_df["fake_probability"]
        .astype(float)
        .to_numpy()
    )

    rows = []

    for threshold in np.arange(
        0.10,
        0.901,
        0.005
    ):
        pred = (
            probs >= threshold
        ).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(
                y_true,
                pred
            ),
            "macro_f1": f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0
            ),
            "fake_precision": precision_score(
                y_true,
                pred,
                zero_division=0
            ),
            "fake_recall": recall_score(
                y_true,
                pred,
                zero_division=0
            ),
            "fake_f1": f1_score(
                y_true,
                pred,
                zero_division=0
            ),
        })

    table = pd.DataFrame(rows)

    table["distance_to_050"] = (
        table["threshold"] - 0.50
    ).abs()

    best = (
        table
        .sort_values(
            [
                "macro_f1",
                "accuracy",
                "distance_to_050"
            ],
            ascending=[
                False,
                False,
                True
            ]
        )
        .iloc[0]
    )

    return (
        float(best["threshold"]),
        table
    )


OPT_THRESHOLD_TTA, threshold_table_tta = (
    find_optimal_threshold_from_oof(
        oof_tta
    )
)

OPT_THRESHOLD_NO_TTA, threshold_table_no = (
    find_optimal_threshold_from_oof(
        oof_no_tta
    )
)

threshold_table_tta.to_csv(
    OUT_DIR / "OOF_threshold_search_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)

threshold_table_no.to_csv(
    OUT_DIR / "OOF_threshold_search_NO_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)


def apply_threshold_report(
    pred_df,
    threshold
):
    y_true = np.array(
        [
            LABEL_MAP[x]
            for x in pred_df["true_label"]
        ],
        dtype=int
    )

    probs = (
        pred_df["fake_probability"]
        .astype(float)
        .to_numpy()
    )

    pred = (
        probs >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            pred
        ),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),
        "real_correct": int(tn),
        "real_total": int(tn + fp),
        "fake_correct": int(tp),
        "fake_total": int(tp + fn),
    }


cal_tta = apply_threshold_report(
    oof_tta,
    OPT_THRESHOLD_TTA
)

cal_no = apply_threshold_report(
    oof_no_tta,
    OPT_THRESHOLD_NO_TTA
)

pd.DataFrame(
    [
        {
            "mode": "TTA",
            **cal_tta
        },
        {
            "mode": "NO_TTA",
            **cal_no
        }
    ]
).to_csv(
    OUT_DIR / "OOF_optimal_threshold_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\\nOOF optimal threshold:"
)

print(
    f"  TTA    : {OPT_THRESHOLD_TTA:.3f} | "
    f"Acc %{cal_tta['accuracy']*100:.2f} | "
    f"Macro-F1 {cal_tta['macro_f1']:.4f} | "
    f"REAL {cal_tta['real_correct']}/{cal_tta['real_total']} | "
    f"FAKE {cal_tta['fake_correct']}/{cal_tta['fake_total']}"
)

print(
    f"  NO-TTA : {OPT_THRESHOLD_NO_TTA:.3f} | "
    f"Acc %{cal_no['accuracy']*100:.2f} | "
    f"Macro-F1 {cal_no['macro_f1']:.4f}"
)


# Alt grup raporu — esas olarak NO-TTA
subgroups = subgroup_table(
    oof_no_tta
)

subgroups.to_csv(
    OUT_DIR / "OOF_subgroup_metrics_NO_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# İNSAN TESTİNDEKİ 24 GÖRÜNTÜNÜN OOF SONUÇLARI
# =========================================================

if HUMAN_MAPPING_CSV.exists():
    human_map = pd.read_csv(
        HUMAN_MAPPING_CSV
    )

    if "master_id" not in human_map.columns:
        print(
            "\nUYARI: human_test_mapping_24.csv içinde "
            "master_id yok; merge yapılmadı."
        )
    else:
        human_map["master_id"] = (
            human_map["master_id"]
            .astype(int)
        )

        human_no_tta = human_map.merge(
            oof_no_tta[
                [
                    "id",
                    "fold",
                    "best_epoch",
                    "true_label",
                    "predicted_label",
                    "fake_probability",
                    "correct",
                ]
            ],
            left_on="master_id",
            right_on="id",
            how="left",
            validate="one_to_one"
        )

        human_tta = human_map.merge(
            oof_tta[
                [
                    "id",
                    "fold",
                    "best_epoch",
                    "true_label",
                    "predicted_label",
                    "fake_probability",
                    "correct",
                ]
            ],
            left_on="master_id",
            right_on="id",
            how="left",
            validate="one_to_one"
        )

        human_no_tta.to_csv(
            OUT_DIR
            / "HUMAN_TEST_24_MODEL_OOF_NO_TTA.csv",
            index=False,
            encoding="utf-8-sig"
        )

        human_tta.to_csv(
            OUT_DIR
            / "HUMAN_TEST_24_MODEL_OOF_TTA.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(
            "\nİnsan testindeki 24 görüntünün "
            "OOF sonuçları ayrıca kaydedildi."
        )


# =========================================================
# GENEL ÖZET DOSYASI
# =========================================================

total_correct_tta = (
    gc_tta["real_correct"]
    +
    gc_tta["fake_correct"]
)

total_correct_no_tta = (
    gc_no_tta["real_correct"]
    +
    gc_no_tta["fake_correct"]
)

summary_lines = [
    "CNN-v4 MultiScale512 | MASTER 408 | Group-Aware 5-Fold OOF",
    "",
    f"Toplam görüntü: 408",
    f"REAL: 204",
    f"FAKE: 204",
    "",
    "TTA:",
    f"  Accuracy : {general_tta['accuracy']:.6f}",
    f"  Precision: {general_tta['precision']:.6f}",
    f"  Recall   : {general_tta['recall']:.6f}",
    f"  F1       : {general_tta['f1']:.6f}",
    f"  Macro-F1 : {general_tta['macro_f1']:.6f}",
    f"  ROC-AUC  : {general_tta['roc_auc']:.6f}",
    f"  Doğru    : {total_correct_tta}/408",
    f"  REAL     : {gc_tta['real_correct']}/{gc_tta['real_total']}",
    f"  FAKE     : {gc_tta['fake_correct']}/{gc_tta['fake_total']}",
    "",
    "NO-TTA:",
    f"  Accuracy : {general_no_tta['accuracy']:.6f}",
    f"  Precision: {general_no_tta['precision']:.6f}",
    f"  Recall   : {general_no_tta['recall']:.6f}",
    f"  F1       : {general_no_tta['f1']:.6f}",
    f"  Macro-F1 : {general_no_tta['macro_f1']:.6f}",
    f"  ROC-AUC  : {general_no_tta['roc_auc']:.6f}",
    f"  Doğru    : {total_correct_no_tta}/408",
    f"  REAL     : {gc_no_tta['real_correct']}/{gc_no_tta['real_total']}",
    f"  FAKE     : {gc_no_tta['fake_correct']}/{gc_no_tta['fake_total']}",
]

(OUT_DIR / "GENEL_OZET.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


# =========================================================
# TERMİNAL
# =========================================================

print(
    "\n"
    +
    "=" * 86
)

print(
    "CNN-v4 MULTISCALE512 | "
    "MASTER408 GROUP-AWARE 5-FOLD TAMAMLANDI"
)

print(
    "=" * 86
)

print(
    "\nNO-TTA OOF:"
)

print(
    f"  Doğru    : "
    f"{total_correct_no_tta}/408 "
    f"(%{general_no_tta['accuracy']*100:.2f})"
)

print(
    f"  REAL     : "
    f"{gc_no_tta['real_correct']}/"
    f"{gc_no_tta['real_total']}"
)

print(
    f"  FAKE     : "
    f"{gc_no_tta['fake_correct']}/"
    f"{gc_no_tta['fake_total']}"
)

print(
    f"  F1       : "
    f"{general_no_tta['f1']:.4f}"
)

print(
    f"  ROC-AUC  : "
    f"{general_no_tta['roc_auc']:.4f}"
)

print(
    "\nTTA OOF:"
)

print(
    f"  Doğru    : "
    f"{total_correct_tta}/408 "
    f"(%{general_tta['accuracy']*100:.2f})"
)

print(
    "\nSonuç klasörü:"
)

print(
    OUT_DIR
)


# =====================================================================
# EK PIPELINE 1 — HUMAN TEST: 24 KATILIMCI vs V4 OOF
# =====================================================================

def human_pairwise_score(model_human_df):
    """
    Google Form eşli bölümündeki 6 soru:
    g1-y1 ... g6-y6.
    Yapay görüntünün FAKE olasılığı daha yüksekse çift doğru.
    """
    rows = []

    for pair_no in range(1, 7):
        g_code = f"g{pair_no}"
        y_code = f"y{pair_no}"

        g = model_human_df[
            model_human_df["human_code"] == g_code
        ].iloc[0]

        y = model_human_df[
            model_human_df["human_code"] == y_code
        ].iloc[0]

        g_prob = float(
            g["fake_probability"]
        )

        y_prob = float(
            y["fake_probability"]
        )

        rows.append({
            "pair_no": pair_no,
            "real_code": g_code,
            "fake_code": y_code,
            "real_fake_probability": g_prob,
            "fake_fake_probability": y_prob,
            "margin_fake_minus_real": (
                y_prob - g_prob
            ),
            "correct": (
                y_prob > g_prob
            )
        })

    out = pd.DataFrame(rows)

    return (
        out,
        int(out["correct"].sum())
    )


def human_single_score(
    model_human_df,
    threshold=0.50
):
    single = model_human_df[
        model_human_df["form_section"]
        .astype(str)
        .str.upper()
        ==
        "TEKIL"
    ].copy()

    true = single["true_label"].map(
        LABEL_MAP
    ).to_numpy(dtype=int)

    probs = single[
        "fake_probability"
    ].astype(float).to_numpy()

    pred = (
        probs >= threshold
    ).astype(int)

    return (
        int((pred == true).sum()),
        len(single)
    )


def create_human_comparison():
    human_dir = (
        OUT_DIR
        / "human_test_comparison"
    )

    human_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    human_tta_path = (
        OUT_DIR
        / "HUMAN_TEST_24_MODEL_OOF_TTA.csv"
    )

    human_no_path = (
        OUT_DIR
        / "HUMAN_TEST_24_MODEL_OOF_NO_TTA.csv"
    )

    if (
        not human_tta_path.exists()
        or
        not human_no_path.exists()
    ):
        print(
            "\\nHuman karşılaştırması atlandı: "
            "model-human OOF dosyaları bulunamadı."
        )
        return

    ht = pd.read_csv(
        human_tta_path
    )

    hn = pd.read_csv(
        human_no_path
    )

    # 24 kişilik Google Form analizinden sabitlenmiş sonuçlar.
    # Araştırmacının kendi deneme yanıtı hariç.
    HUMAN_N = 24
    HUMAN_PAIR_ACC = 58.3
    HUMAN_SINGLE_ACC = 54.5
    HUMAN_18_ACC = 55.8
    HUMAN_24_EQ_ACC = 56.4

    # Görsel-bazlı insan doğrulukları (%)
    human_image_acc = {
        "g1": 66.7,
        "g2": 37.5,
        "g3": 75.0,
        "g4": 58.3,
        "g5": 50.0,
        "g6": 62.5,
        "g7": 70.8,
        "g8": 54.2,
        "g9": 66.7,
        "g10": 79.2,
        "g11": 33.3,
        "g12": 58.3,
        "y1": 66.7,
        "y2": 37.5,
        "y3": 75.0,
        "y4": 58.3,
        "y5": 50.0,
        "y6": 62.5,
        "y13": 58.3,
        "y14": 62.5,
        "y15": 33.3,
        "y16": 29.2,
        "y17": 45.8,
        "y18": 62.5,
    }

    # Eşli 6 soru insan doğrulukları
    human_pair_acc = {
        1: 66.7,
        2: 37.5,
        3: 75.0,
        4: 58.3,
        5: 50.0,
        6: 62.5,
    }

    # -----------------------------
    # Model aynı 18 görevde
    # -----------------------------
    pair_tta_df, pair_tta_correct = (
        human_pairwise_score(ht)
    )

    pair_no_df, pair_no_correct = (
        human_pairwise_score(hn)
    )

    single_tta_correct, single_total = (
        human_single_score(
            ht,
            threshold=0.50
        )
    )

    single_no_correct, _ = (
        human_single_score(
            hn,
            threshold=0.50
        )
    )

    single_tta_cal_correct, _ = (
        human_single_score(
            ht,
            threshold=OPT_THRESHOLD_TTA
        )
    )

    model18_tta = (
        pair_tta_correct
        +
        single_tta_correct
    )

    model18_no = (
        pair_no_correct
        +
        single_no_correct
    )

    model18_tta_cal = (
        pair_tta_correct
        +
        single_tta_cal_correct
    )

    # 24 görüntüyü bağımsız olarak tek tek model sınıflandırması
    def individual24(df_model, threshold):
        true = df_model[
            "true_label"
        ].map(
            LABEL_MAP
        ).to_numpy(dtype=int)

        probs = df_model[
            "fake_probability"
        ].astype(float).to_numpy()

        pred = (
            probs >= threshold
        ).astype(int)

        return int(
            (pred == true).sum()
        )

    ind24_tta = individual24(
        ht,
        0.50
    )

    ind24_tta_cal = individual24(
        ht,
        OPT_THRESHOLD_TTA
    )

    ind24_no = individual24(
        hn,
        0.50
    )

    summary = pd.DataFrame([
        {
            "measurement": "Eşli 6 karar",
            "human_pct": HUMAN_PAIR_ACC,
            "v4_tta_pct": 100 * pair_tta_correct / 6,
            "v4_no_tta_pct": 100 * pair_no_correct / 6,
            "v4_tta_oof_threshold_pct": 100 * pair_tta_correct / 6,
            "note": "Pairwise görev threshold kullanmaz."
        },
        {
            "measurement": "Tekil 12 karar",
            "human_pct": HUMAN_SINGLE_ACC,
            "v4_tta_pct": 100 * single_tta_correct / 12,
            "v4_no_tta_pct": 100 * single_no_correct / 12,
            "v4_tta_oof_threshold_pct": 100 * single_tta_cal_correct / 12,
            "note": "İnsanla doğrudan aynı tekil görev."
        },
        {
            "measurement": "Genel 18 karar",
            "human_pct": HUMAN_18_ACC,
            "v4_tta_pct": 100 * model18_tta / 18,
            "v4_no_tta_pct": 100 * model18_no / 18,
            "v4_tta_oof_threshold_pct": 100 * model18_tta_cal / 18,
            "note": "Ana insan-model karşılaştırması."
        },
        {
            "measurement": "Model 24 görüntü tek tek",
            "human_pct": np.nan,
            "v4_tta_pct": 100 * ind24_tta / 24,
            "v4_no_tta_pct": 100 * ind24_no / 24,
            "v4_tta_oof_threshold_pct": 100 * ind24_tta_cal / 24,
            "note": "Eşli bölüm insanlarla aynı ölçüm değildir."
        },
    ])

    summary.to_csv(
        human_dir
        / "human_vs_v4_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Pair detayları
    pair_tta_df[
        "human_pair_accuracy_pct"
    ] = pair_tta_df[
        "pair_no"
    ].map(human_pair_acc)

    pair_tta_df.to_csv(
        human_dir
        / "human_vs_v4_pairwise_6_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Görsel bazlı detay
    detail = ht[
        [
            "human_code",
            "form_section",
            "true_label",
            "predicted_label",
            "fake_probability",
            "correct"
        ]
    ].copy()

    detail[
        "human_accuracy_pct"
    ] = detail[
        "human_code"
    ].map(human_image_acc)

    detail.to_csv(
        human_dir
        / "human_vs_v4_24_image_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------
    # Grafik 1: 18 karar
    # -----------------------------
    cats = [
        "Eşli\\n6 karar",
        "Tekil\\n12 karar",
        "Genel\\n18 karar"
    ]

    human_vals = [
        HUMAN_PAIR_ACC,
        HUMAN_SINGLE_ACC,
        HUMAN_18_ACC
    ]

    v4_vals = [
        100 * pair_tta_correct / 6,
        100 * single_tta_correct / 12,
        100 * model18_tta / 18
    ]

    x = np.arange(
        len(cats)
    )

    w = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    b1 = ax.bar(
        x - w/2,
        human_vals,
        width=w,
        label=f"İnsan ortalaması (n={HUMAN_N})"
    )

    b2 = ax.bar(
        x + w/2,
        v4_vals,
        width=w,
        label="CNN-v4 TTA"
    )

    ax.set_xticks(
        x,
        cats
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "Doğruluk (%)"
    )

    ax.set_title(
        "İnsan vs CNN-v4 — Aynı Görev Üzerinde"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        b1,
        fmt="%.1f%%",
        padding=3
    )

    ax.bar_label(
        b2,
        fmt="%.1f%%",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        human_dir
        / "01_human_vs_v4_18_task.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Grafik 2: tekil 12 görüntü
    # -----------------------------
    single_detail = detail[
        detail["form_section"]
        .astype(str)
        .str.upper()
        ==
        "TEKIL"
    ].copy()

    single_detail[
        "model_correct_pct"
    ] = (
        single_detail["correct"]
        .astype(float)
        * 100
    )

    x = np.arange(
        len(single_detail)
    )

    w = 0.36

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.bar(
        x - w/2,
        single_detail[
            "human_accuracy_pct"
        ],
        width=w,
        label="İnsan doğruluğu"
    )

    ax.bar(
        x + w/2,
        single_detail[
            "model_correct_pct"
        ],
        width=w,
        label="CNN-v4 TTA"
    )

    ax.set_xticks(
        x,
        single_detail[
            "human_code"
        ]
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "Doğruluk (%)"
    )

    ax.set_title(
        "Tekil 12 Görüntü — İnsan ve CNN-v4"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        human_dir
        / "02_human_vs_v4_single12.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Grafik 3: insanların en zorlandığı görseller
    # -----------------------------
    sorted_detail = detail.sort_values(
        "human_accuracy_pct",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    x = np.arange(
        len(sorted_detail)
    )

    ax.bar(
        x,
        sorted_detail[
            "human_accuracy_pct"
        ]
    )

    ax.set_xticks(
        x,
        sorted_detail[
            "human_code"
        ],
        rotation=45
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "İnsan doğruluğu (%)"
    )

    ax.set_title(
        "İnsanların Zorlandığı Görseller ve V4 Sonucu"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for i, (_, r) in enumerate(
        sorted_detail.iterrows()
    ):
        mark = (
            "V4 ✓"
            if bool(r["correct"])
            else "V4 ✗"
        )

        ax.text(
            i,
            float(r["human_accuracy_pct"]) + 2,
            mark,
            ha="center",
            fontsize=8
        )

    plt.tight_layout()

    plt.savefig(
        human_dir
        / "03_human_image_difficulty_vs_v4.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    human_report = [
        "CNN-v4 HUMAN TEST KARŞILAŞTIRMASI",
        "",
        f"Katılımcı: {HUMAN_N}",
        f"İnsan eşli: %{HUMAN_PAIR_ACC:.1f}",
        f"V4 TTA eşli: {pair_tta_correct}/6 = %{100*pair_tta_correct/6:.1f}",
        f"İnsan tekil: %{HUMAN_SINGLE_ACC:.1f}",
        f"V4 TTA tekil: {single_tta_correct}/12 = %{100*single_tta_correct/12:.1f}",
        f"İnsan 18 karar: %{HUMAN_18_ACC:.1f}",
        f"V4 TTA 18 karar: {model18_tta}/18 = %{100*model18_tta/18:.1f}",
        f"V4 TTA 24 görüntü tek tek: {ind24_tta}/24 = %{100*ind24_tta/24:.1f}",
        f"OOF threshold={OPT_THRESHOLD_TTA:.3f} ile V4 TTA 24 tek tek: "
        f"{ind24_tta_cal}/24 = %{100*ind24_tta_cal/24:.1f}",
        "",
        "Not: %18-görev skoru modelin genel 408 doğruluğu değildir."
    ]

    (
        human_dir
        / "HUMAN_V4_RAPOR.txt"
    ).write_text(
        "\\n".join(
            human_report
        ),
        encoding="utf-8"
    )

    print(
        "\\nHUMAN TEST V4:"
    )

    print(
        f"  İnsan 18 karar ort.: %{HUMAN_18_ACC:.1f}"
    )

    print(
        f"  V4 TTA 18 karar     : "
        f"{model18_tta}/18 "
        f"(%{100*model18_tta/18:.1f})"
    )

    print(
        f"  V4 TTA 24 tek tek   : "
        f"{ind24_tta}/24 "
        f"(%{100*ind24_tta/24:.1f})"
    )


# =====================================================================
# EK PIPELINE 2 — 24 BAĞIMSIZ DIŞ TEST / 5-FOLD ENSEMBLE
# =====================================================================

def resolve_external_test_path(raw_path):
    raw = str(
        raw_path
    ).strip()

    direct = Path(raw)

    if direct.exists():
        return direct.resolve()

    wp = PureWindowsPath(raw)

    candidate = (
        BASE
        / "test"
        / wp.name
    )

    if candidate.exists():
        return candidate.resolve()

    matches = [
        p
        for p in (
            BASE / "test"
        ).rglob(wp.name)
        if p.is_file()
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Bağımsız test görüntüsü bulunamadı: {raw}"
    )


def external_metric(
    true,
    probs,
    threshold=0.50
):
    true = np.asarray(
        true,
        dtype=int
    )

    probs = np.asarray(
        probs,
        dtype=float
    )

    pred = (
        probs >= threshold
    ).astype(int)

    cm = confusion_matrix(
        true,
        pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            true,
            pred
        ),
        "precision": precision_score(
            true,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            true,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            true,
            pred,
            zero_division=0
        ),
        "macro_f1": f1_score(
            true,
            pred,
            average="macro",
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            true,
            probs
        ),
        "real_correct": int(tn),
        "real_total": int(tn + fp),
        "fake_correct": int(tp),
        "fake_total": int(tp + fn),
        "pred": pred,
        "cm": cm,
    }


def external_pairwise(
    test_df,
    prob_col
):
    rows = []

    for pair_no, g in test_df.groupby(
        "pair_no",
        sort=True
    ):
        real = g[
            g["label"] == "REAL"
        ].iloc[0]

        fake = g[
            g["label"] == "FAKE"
        ].iloc[0]

        real_prob = float(
            real[prob_col]
        )

        fake_prob = float(
            fake[prob_col]
        )

        rows.append({
            "pair_no": int(pair_no),
            "real_file": real["file_name"],
            "fake_file": fake["file_name"],
            "real_fake_probability": real_prob,
            "fake_fake_probability": fake_prob,
            "margin_fake_minus_real": (
                fake_prob - real_prob
            ),
            "correct": (
                fake_prob > real_prob
            )
        })

    out = pd.DataFrame(
        rows
    )

    return (
        out,
        int(out["correct"].sum())
    )


@torch.no_grad()
def predict_tensor_with_model(
    model,
    tensor,
    tta=False
):
    model.eval()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        logits_normal = model(
            tensor
        )

        if tta:
            logits_flip = model(
                torch.flip(
                    tensor,
                    dims=[3]
                )
            )

            logits = (
                logits_normal
                +
                logits_flip
            ) / 2.0
        else:
            logits = logits_normal

    return (
        torch.sigmoid(
            logits.float()
        )
        .detach()
        .cpu()
        .numpy()
    )


def plot_external_confusion(
    cm,
    title,
    path
):
    fig, ax = plt.subplots(
        figsize=(6.5, 5.5)
    )

    im = ax.imshow(
        cm
    )

    ax.set_xticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    ax.set_yticks(
        [0, 1],
        ["REAL", "FAKE"]
    )

    ax.set_xlabel(
        "Tahmin"
    )

    ax.set_ylabel(
        "Gerçek"
    )

    ax.set_title(
        title
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def run_external_test_24():
    test_csv = (
        BASE
        / "test"
        / "test_dataset_pairli.csv"
    )

    if not test_csv.exists():
        print(
            "\\nBağımsız 24 test atlandı: "
            "test/test_dataset_pairli.csv bulunamadı."
        )
        return

    ext_dir = (
        OUT_DIR
        / "independent_external_test_24"
    )

    ext_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw = pd.read_csv(
        test_csv
    )

    required = {
        "no",
        "file_name",
        "file_path",
        "label",
        "format",
        "generator"
    }

    missing = (
        required
        -
        set(raw.columns)
    )

    if missing:
        raise RuntimeError(
            "Bağımsız test CSV eksik sütunlar: "
            + ", ".join(
                sorted(missing)
            )
        )

    raw["label"] = (
        raw["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    raw["pair_no"] = pd.to_numeric(
        raw["no"],
        errors="raise"
    ).astype(int)

    raw["file_path_resolved"] = [
        str(
            resolve_external_test_path(
                p
            )
        )
        for p
        in raw["file_path"]
    ]

    if len(raw) != 24:
        raise RuntimeError(
            f"Bağımsız test 24 görüntü olmalı. Bulunan: {len(raw)}"
        )

    if (
        (raw["label"] == "REAL").sum()
        !=
        12
    ):
        raise RuntimeError(
            "Bağımsız test REAL sayısı 12 değil."
        )

    if (
        (raw["label"] == "FAKE").sum()
        !=
        12
    ):
        raise RuntimeError(
            "Bağımsız test FAKE sayısı 12 değil."
        )

    pair_sizes = (
        raw.groupby(
            "pair_no"
        )
        .size()
    )

    if (
        len(pair_sizes) != 12
        or
        not (
            pair_sizes == 2
        ).all()
    ):
        raise RuntimeError(
            "Bağımsız testte 12 çift ve her çiftte 2 görüntü olmalı."
        )

    for pair_no, g in raw.groupby(
        "pair_no"
    ):
        if set(
            g["label"]
        ) != {
            "REAL",
            "FAKE"
        }:
            raise RuntimeError(
                f"Pair {pair_no} REAL+FAKE içermiyor."
            )

    # Görüntüler
    print(
        "\\n24 bağımsız test görüntüsü yükleniyor..."
    )

    ext_X = torch.stack(
        [
            load_rgb(
                p
            )
            for p
            in raw[
                "file_path_resolved"
            ]
        ]
    ).to(
        DEVICE
    )

    y_true = (
        raw["label"]
        .map(
            LABEL_MAP
        )
        .to_numpy(dtype=int)
    )

    probs_tta = []
    probs_no = []
    fold_rows = []

    for fold in range(
        1,
        6
    ):
        fold_dir = (
            OUT_DIR
            / f"fold_{fold}"
        )

        ckpts = list(
            fold_dir.glob(
                "BEST_MODEL_epoch*.pth"
            )
        )

        if len(ckpts) != 1:
            raise RuntimeError(
                f"V4 Fold {fold} checkpoint bulunamadı."
            )

        try:
            state = torch.load(
                ckpts[0],
                map_location="cpu",
                weights_only=True
            )
        except TypeError:
            state = torch.load(
                ckpts[0],
                map_location="cpu"
            )

        model = V4MultiScale512().to(
            DEVICE
        )

        model.load_state_dict(
            state
        )

        p_no = predict_tensor_with_model(
            model,
            ext_X,
            tta=False
        )

        p_tta = predict_tensor_with_model(
            model,
            ext_X,
            tta=True
        )

        probs_no.append(
            p_no
        )

        probs_tta.append(
            p_tta
        )

        m_no = external_metric(
            y_true,
            p_no,
            0.50
        )

        m_tta = external_metric(
            y_true,
            p_tta,
            0.50
        )

        fold_rows.append({
            "fold": fold,
            "checkpoint": ckpts[0].name,
            "accuracy_tta": m_tta["accuracy"],
            "real_correct_tta": m_tta["real_correct"],
            "fake_correct_tta": m_tta["fake_correct"],
            "macro_f1_tta": m_tta["macro_f1"],
            "roc_auc_tta": m_tta["roc_auc"],
            "accuracy_no_tta": m_no["accuracy"],
            "real_correct_no_tta": m_no["real_correct"],
            "fake_correct_no_tta": m_no["fake_correct"],
            "macro_f1_no_tta": m_no["macro_f1"],
            "roc_auc_no_tta": m_no["roc_auc"],
        })

        print(
            f"  Fold {fold} | "
            f"TTA %{100*m_tta['accuracy']:.2f} | "
            f"NO-TTA %{100*m_no['accuracy']:.2f}"
        )

        del model
        torch.cuda.empty_cache()

    P_TTA = np.stack(
        probs_tta,
        axis=0
    )

    P_NO = np.stack(
        probs_no,
        axis=0
    )

    ens_tta = P_TTA.mean(
        axis=0
    )

    ens_no = P_NO.mean(
        axis=0
    )

    std_tta = P_TTA.std(
        axis=0
    )

    # 0.50 metrik
    ens_tta_050 = external_metric(
        y_true,
        ens_tta,
        0.50
    )

    ens_no_050 = external_metric(
        y_true,
        ens_no,
        0.50
    )

    # 408 OOF'tan bulunan threshold ile dış test
    ens_tta_cal = external_metric(
        y_true,
        ens_tta,
        OPT_THRESHOLD_TTA
    )

    ens_no_cal = external_metric(
        y_true,
        ens_no,
        OPT_THRESHOLD_NO_TTA
    )

    result = raw.copy()

    for i in range(5):
        result[
            f"fold_{i+1}_fake_prob_tta"
        ] = P_TTA[i]

        result[
            f"fold_{i+1}_fake_prob_no_tta"
        ] = P_NO[i]

    result[
        "ensemble_fake_probability_tta"
    ] = ens_tta

    result[
        "ensemble_fake_probability_no_tta"
    ] = ens_no

    result[
        "fold_std_tta"
    ] = std_tta

    result[
        "prediction_tta_050"
    ] = [
        LABEL_NAME[x]
        for x in ens_tta_050[
            "pred"
        ]
    ]

    result[
        "prediction_tta_oof_threshold"
    ] = [
        LABEL_NAME[x]
        for x in ens_tta_cal[
            "pred"
        ]
    ]

    result[
        "correct_tta_050"
    ] = (
        result[
            "prediction_tta_050"
        ]
        ==
        result["label"]
    )

    result[
        "correct_tta_oof_threshold"
    ] = (
        result[
            "prediction_tta_oof_threshold"
        ]
        ==
        result["label"]
    )

    pair_tta_df, pair_tta_correct = (
        external_pairwise(
            result,
            "ensemble_fake_probability_tta"
        )
    )

    pair_no_df, pair_no_correct = (
        external_pairwise(
            result,
            "ensemble_fake_probability_no_tta"
        )
    )

    result.to_csv(
        ext_dir
        / "01_independent24_v4_ensemble_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pair_tta_df.to_csv(
        ext_dir
        / "02_independent24_pairwise_TTA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        ext_dir
        / "03_fold_by_fold_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metric_rows = []

    for mode, m in [
        ("TTA_0.50", ens_tta_050),
        (
            f"TTA_OOF_threshold_{OPT_THRESHOLD_TTA:.3f}",
            ens_tta_cal
        ),
        ("NO_TTA_0.50", ens_no_050),
        (
            f"NO_TTA_OOF_threshold_{OPT_THRESHOLD_NO_TTA:.3f}",
            ens_no_cal
        ),
    ]:
        metric_rows.append({
            "mode": mode,
            "threshold": m["threshold"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall_fake": m["recall"],
            "f1_fake": m["f1"],
            "macro_f1": m["macro_f1"],
            "roc_auc": m["roc_auc"],
            "real_correct": m["real_correct"],
            "real_total": m["real_total"],
            "fake_correct": m["fake_correct"],
            "fake_total": m["fake_total"],
        })

    pd.DataFrame(
        metric_rows
    ).to_csv(
        ext_dir
        / "04_independent24_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # -----------------------------
    # Grafik: Fold + ensemble
    # -----------------------------
    fold_df = pd.DataFrame(
        fold_rows
    )

    labels = [
        "Fold 1",
        "Fold 2",
        "Fold 3",
        "Fold 4",
        "Fold 5",
        "Ensemble"
    ]

    vals_tta = (
        (
            100
            *
            fold_df[
                "accuracy_tta"
            ].to_numpy(
                dtype=float
            )
        ).tolist()
        +
        [
            100
            *
            ens_tta_050[
                "accuracy"
            ]
        ]
    )

    vals_no = (
        (
            100
            *
            fold_df[
                "accuracy_no_tta"
            ].to_numpy(
                dtype=float
            )
        ).tolist()
        +
        [
            100
            *
            ens_no_050[
                "accuracy"
            ]
        ]
    )

    x = np.arange(
        6
    )

    w = 0.36

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    b1 = ax.bar(
        x - w/2,
        vals_tta,
        width=w,
        label="TTA"
    )

    b2 = ax.bar(
        x + w/2,
        vals_no,
        width=w,
        label="NO-TTA"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "Doğruluk (%)"
    )

    ax.set_title(
        "CNN-v4 — 24 Bağımsız Test: 5 Fold ve Ensemble"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        b1,
        fmt="%.1f%%",
        padding=3
    )

    ax.bar_label(
        b2,
        fmt="%.1f%%",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        ext_dir
        / "05_fold_vs_ensemble.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Grafik: 0.50 vs OOF threshold
    # -----------------------------
    labels = [
        "TTA\\n0.50",
        f"TTA\\nOOF th={OPT_THRESHOLD_TTA:.3f}",
        "NO-TTA\\n0.50",
        f"NO-TTA\\nOOF th={OPT_THRESHOLD_NO_TTA:.3f}"
    ]

    vals = [
        100 * ens_tta_050["accuracy"],
        100 * ens_tta_cal["accuracy"],
        100 * ens_no_050["accuracy"],
        100 * ens_no_cal["accuracy"]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    bars = ax.bar(
        labels,
        vals
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "Doğruluk (%)"
    )

    ax.set_title(
        "Bağımsız 24 Test — Sabit 0.50 ve OOF Kalibre Threshold"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        ext_dir
        / "06_threshold_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    plot_external_confusion(
        ens_tta_050["cm"],
        (
            "CNN-v4 5-Fold Ensemble TTA — Threshold 0.50\\n"
            f"Accuracy %{100*ens_tta_050['accuracy']:.1f}"
        ),
        ext_dir
        / "07_confusion_TTA_050.png"
    )

    plot_external_confusion(
        ens_tta_cal["cm"],
        (
            "CNN-v4 5-Fold Ensemble TTA — OOF Threshold\\n"
            f"Threshold={OPT_THRESHOLD_TTA:.3f} | "
            f"Accuracy %{100*ens_tta_cal['accuracy']:.1f}"
        ),
        ext_dir
        / "08_confusion_TTA_OOF_threshold.png"
    )

    # -----------------------------
    # Grafik: REAL / FAKE
    # -----------------------------
    labels = [
        "REAL",
        "FAKE"
    ]

    v050 = [
        100
        *
        ens_tta_050[
            "real_correct"
        ]
        /
        12,

        100
        *
        ens_tta_050[
            "fake_correct"
        ]
        /
        12
    ]

    vcal = [
        100
        *
        ens_tta_cal[
            "real_correct"
        ]
        /
        12,

        100
        *
        ens_tta_cal[
            "fake_correct"
        ]
        /
        12
    ]

    x = np.arange(
        2
    )

    w = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    b1 = ax.bar(
        x - w/2,
        v050,
        width=w,
        label="TTA threshold 0.50"
    )

    b2 = ax.bar(
        x + w/2,
        vcal,
        width=w,
        label=f"TTA OOF threshold {OPT_THRESHOLD_TTA:.3f}"
    )

    ax.set_xticks(
        x,
        labels
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "Doğru sınıflandırma (%)"
    )

    ax.set_title(
        "Bağımsız Test — REAL / FAKE Başarısı"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        b1,
        fmt="%.1f%%",
        padding=3
    )

    ax.bar_label(
        b2,
        fmt="%.1f%%",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        ext_dir
        / "09_real_fake_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Grafik: pairwise 12
    # -----------------------------
    margins = (
        100
        *
        pair_tta_df[
            "margin_fake_minus_real"
        ].to_numpy()
    )

    x = np.arange(
        1,
        13
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.bar(
        x,
        margins
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.5
    )

    ax.set_xticks(
        x,
        [
            f"Çift {i}"
            for i in x
        ]
    )

    ax.set_ylabel(
        "FAKE olasılık farkı\\n(Yapay - Gerçek, yüzde puan)"
    )

    ax.set_title(
        "Bağımsız 12 Çift — CNN-v4 Pairwise TTA"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        ext_dir
        / "10_pairwise_12_margin.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------
    # Grafik: format
    # -----------------------------
    format_rows = []

    for fmt, g in result.groupby(
        "format"
    ):
        format_rows.append({
            "format": fmt,
            "n": len(g),
            "accuracy": g[
                "correct_tta_050"
            ].mean()
        })

    format_df = pd.DataFrame(
        format_rows
    )

    format_df.to_csv(
        ext_dir
        / "11_format_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(format_df):
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        bars = ax.bar(
            format_df[
                "format"
            ],
            100
            *
            format_df[
                "accuracy"
            ]
        )

        ax.set_ylim(
            0,
            105
        )

        ax.set_ylabel(
            "TTA doğruluğu (%)"
        )

        ax.set_title(
            "Bağımsız Test — Format Bazlı V4"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3
        )

        plt.tight_layout()

        plt.savefig(
            ext_dir
            / "11_format_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # Generator
    fake_result = result[
        result["label"] == "FAKE"
    ]

    gen_rows = []

    for gen, g in fake_result.groupby(
        "generator"
    ):
        gen_rows.append({
            "generator": gen,
            "n": len(g),
            "fake_detection_accuracy": g[
                "correct_tta_050"
            ].mean()
        })

    gen_df = pd.DataFrame(
        gen_rows
    )

    gen_df.to_csv(
        ext_dir
        / "12_generator_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if len(gen_df):
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        bars = ax.bar(
            gen_df[
                "generator"
            ],
            100
            *
            gen_df[
                "fake_detection_accuracy"
            ]
        )

        ax.set_ylim(
            0,
            105
        )

        ax.set_ylabel(
            "FAKE tespit doğruluğu (%)"
        )

        ax.set_title(
            "Bağımsız Test — Generator Bazlı V4"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        ax.bar_label(
            bars,
            fmt="%.1f%%",
            padding=3
        )

        plt.tight_layout()

        plt.savefig(
            ext_dir
            / "12_generator_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    report = [
        "CNN-v4 — 24 BAĞIMSIZ DIŞ TEST / 5-FOLD ENSEMBLE",
        "",
        "TTA threshold 0.50:",
        f"  Doğru: {ens_tta_050['real_correct'] + ens_tta_050['fake_correct']}/24 "
        f"(%{100*ens_tta_050['accuracy']:.2f})",
        f"  REAL: {ens_tta_050['real_correct']}/12",
        f"  FAKE: {ens_tta_050['fake_correct']}/12",
        f"  Macro-F1: {ens_tta_050['macro_f1']:.4f}",
        f"  ROC-AUC: {ens_tta_050['roc_auc']:.4f}",
        f"  Pairwise: {pair_tta_correct}/12 "
        f"(%{100*pair_tta_correct/12:.2f})",
        "",
        f"TTA OOF threshold = {OPT_THRESHOLD_TTA:.3f}:",
        f"  Doğru: {ens_tta_cal['real_correct'] + ens_tta_cal['fake_correct']}/24 "
        f"(%{100*ens_tta_cal['accuracy']:.2f})",
        f"  REAL: {ens_tta_cal['real_correct']}/12",
        f"  FAKE: {ens_tta_cal['fake_correct']}/12",
        "",
        "NO-TTA threshold 0.50:",
        f"  Doğru: {ens_no_050['real_correct'] + ens_no_050['fake_correct']}/24 "
        f"(%{100*ens_no_050['accuracy']:.2f})",
        f"  REAL: {ens_no_050['real_correct']}/12",
        f"  FAKE: {ens_no_050['fake_correct']}/12",
        f"  Pairwise: {pair_no_correct}/12 "
        f"(%{100*pair_no_correct/12:.2f})",
        "",
        "Not: OOF threshold yalnızca 408 OOF tahminlerinden seçildi; "
        "bağımsız 24 sete bakılarak optimize edilmedi."
    ]

    (
        ext_dir
        / "INDEPENDENT24_V4_RAPOR.txt"
    ).write_text(
        "\\n".join(
            report
        ),
        encoding="utf-8"
    )

    print(
        "\\nBAĞIMSIZ 24 — V4 5-FOLD ENSEMBLE:"
    )

    print(
        f"  TTA 0.50 : "
        f"{ens_tta_050['real_correct'] + ens_tta_050['fake_correct']}/24 "
        f"(%{100*ens_tta_050['accuracy']:.2f}) | "
        f"REAL {ens_tta_050['real_correct']}/12 | "
        f"FAKE {ens_tta_050['fake_correct']}/12 | "
        f"PAIR {pair_tta_correct}/12"
    )

    print(
        f"  TTA OOF th={OPT_THRESHOLD_TTA:.3f}: "
        f"{ens_tta_cal['real_correct'] + ens_tta_cal['fake_correct']}/24 "
        f"(%{100*ens_tta_cal['accuracy']:.2f}) | "
        f"REAL {ens_tta_cal['real_correct']}/12 | "
        f"FAKE {ens_tta_cal['fake_correct']}/12"
    )

    print(
        f"  NO-TTA 0.50: "
        f"{ens_no_050['real_correct'] + ens_no_050['fake_correct']}/24 "
        f"(%{100*ens_no_050['accuracy']:.2f})"
    )

    del ext_X
    torch.cuda.empty_cache()


# =====================================================================
# EK PIPELINE 3 — V3 BASELINE vs V4 OOF GRAFİĞİ
# =====================================================================

def compare_v3_v4():
    v3_dir = (
        BASE
        / "cross_validation_master408"
        / "cnn_v3_stronger_master408"
    )

    v3_tta_path = (
        v3_dir
        / "OOF_ALL_408_TTA.csv"
    )

    v3_no_path = (
        v3_dir
        / "OOF_ALL_408_NO_TTA.csv"
    )

    if (
        not v3_tta_path.exists()
        or
        not v3_no_path.exists()
    ):
        print(
            "\\nV3-V4 grafik karşılaştırması atlandı: "
            "V3 OOF dosyaları bulunamadı."
        )
        return

    def csv_accuracy(path):
        d = pd.read_csv(
            path
        )

        c = d[
            "correct"
        ]

        if c.dtype != bool:
            c = (
                c.astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False
                })
            )

        return float(
            c.mean()
        )

    v3_tta_acc = csv_accuracy(
        v3_tta_path
    )

    v3_no_acc = csv_accuracy(
        v3_no_path
    )

    compare = pd.DataFrame([
        {
            "model": "V3 Stronger",
            "TTA_accuracy": v3_tta_acc,
            "NO_TTA_accuracy": v3_no_acc
        },
        {
            "model": "V4 MultiScale512",
            "TTA_accuracy": general_tta[
                "accuracy"
            ],
            "NO_TTA_accuracy": general_no_tta[
                "accuracy"
            ]
        }
    ])

    compare.to_csv(
        OUT_DIR
        / "V3_vs_V4_OOF_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    x = np.arange(
        2
    )

    w = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    b1 = ax.bar(
        x - w/2,
        100
        *
        compare[
            "TTA_accuracy"
        ],
        width=w,
        label="TTA"
    )

    b2 = ax.bar(
        x + w/2,
        100
        *
        compare[
            "NO_TTA_accuracy"
        ],
        width=w,
        label="NO-TTA"
    )

    ax.set_xticks(
        x,
        compare[
            "model"
        ]
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_ylabel(
        "408 OOF doğruluğu (%)"
    )

    ax.set_title(
        "CNN-v3 vs CNN-v4 — Aynı 408 ve Aynı Group-Aware 5-Fold"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.bar_label(
        b1,
        fmt="%.1f%%",
        padding=3
    )

    ax.bar_label(
        b2,
        fmt="%.1f%%",
        padding=3
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR
        / "V3_vs_V4_OOF_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\\nV3 vs V4 OOF:"
    )

    print(
        f"  V3 TTA: %{100*v3_tta_acc:.2f}"
    )

    print(
        f"  V4 TTA: %{100*general_tta['accuracy']:.2f}"
    )


# =====================================================================
# FULL PIPELINE SON AŞAMALAR
# =====================================================================

create_human_comparison()
run_external_test_24()
compare_v3_v4()

print(
    "\\n"
    +
    "=" * 86
)

print(
    "CNN-v4 FULL PIPELINE TAMAMLANDI"
)

print(
    "=" * 86
)

print(
    "\\nV4 MİMARİ:"
)

print(
    "  BatchNorm + Residual/ECA 64→128→256 + "
    "Light Bottleneck512 + ECA+Spatial Attention + "
    "Multi-Scale Fusion(128/256/512) + 1792→256→64→1"
)

print(
    f"  Label smoothing: {LABEL_SMOOTH}"
)

print(
    f"  OOF TTA: %{general_tta['accuracy']*100:.2f}"
)

print(
    f"  OOF NO-TTA: %{general_no_tta['accuracy']*100:.2f}"
)

print(
    f"  OOF optimal TTA threshold: {OPT_THRESHOLD_TTA:.3f}"
)

print(
    "\\nTüm sonuçlar:"
)

print(
    OUT_DIR
)
