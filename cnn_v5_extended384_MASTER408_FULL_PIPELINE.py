# -*- coding: utf-8 -*-
"""
CNN-v5 EXTENDED384 — MASTER 408 / GROUP-AWARE 5-FOLD / FULL PIPELINE

Bu sürüm CNN-v3 Stronger'ı temel alır. V3'ün 64→128→256 Residual+ECA
gövdesi korunur; yalnızca 384-kanal yeni stage eklenir ve classifier
3072→256→64→1 olacak şekilde genişletilir.

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
- Eğitim yalnızca master 408 üzerinde yapılır.
- Eğitim bittikten sonra bağımsız 24 test, human test ve sentetik noise testi otomatik çalışır.
- Noise yalnızca TEST aşamasında eklenir; eğitim verisine eklenmez.
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

OUT_DIR = BASE / "cross_validation_master408" / "cnn_v5_extended384"

SIZE = 320
BATCH = 8

MAX_EPOCHS = 230
PATIENCE = 80

LR = 0.0001
MIN_LR = 0.000001
WEIGHT_DECAY = 0.0003

LABEL_SMOOTH = 0.05
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

        y = self.relu(y + old)

        return y


# =========================================================
# CNN-v5 EXTENDED384
#
# V3 STRONGER'DAN KORUNAN:
# - 320x320 RGB giriş
# - BatchNorm
# - Residual + ECA 64
# - Residual + ECA 128
# - Residual + ECA 256
# - Adaptive AvgPool2x2 + MaxPool2x2
# - Dropout 0.50 / 0.25
# - AdamW / LR / weight decay
# - Label smoothing 0.05
# - HFlip + ±8 px shift
# - Group-aware 5-fold
# - TTA
#
# V5'TE YENİ:
# 1) V3'ün doğal devamı olarak 256 -> 384 downsample stage
# 2) 384 kanalda aynı tip Residual + ECA bloğu
# 3) Pooling sonrası 3072 özellik
# 4) Classifier 3072 -> 256 -> 64 -> 1
#
# Yani V5, V3'ün yapısını bozmadan bir stage büyütülmüş halidir.
# =========================================================

class V5Extended384(nn.Module):
    def __init__(self):
        super().__init__()

        # V3 ile aynı
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

        self.block64 = ResidualECABlock(64)

        # V3 ile aynı
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

        self.block128 = ResidualECABlock(128)

        # V3 ile aynı
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

        self.block256 = ResidualECABlock(256)

        # V5'te tek yeni feature stage:
        # V3'teki downsample mantığının aynısı
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

        self.block384 = ResidualECABlock(384)

        # V3 pooling mantığı aynen korunuyor
        self.avg2 = nn.AdaptiveAvgPool2d((2, 2))
        self.max2 = nn.AdaptiveMaxPool2d((2, 2))

        # 384 * 2 * 2 = 1536 (Avg)
        # 384 * 2 * 2 = 1536 (Max)
        # toplam = 3072
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
        "CNN-v5 Extended384 | "
        f"Master408 Group-Aware Fold {fold} | "
        f"Best={best_epoch}, Stop={stopped_epoch}",
        fontsize=14
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.savefig(
        folder
        / f"fold_{fold}_CNN_V5_EXTENDED384_OZET.png",
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

temp_model = V5Extended384()

param_count = sum(
    p.numel()
    for p in temp_model.parameters()
    if p.requires_grad
)

del temp_model

print(
    "\nCNN-v5 Extended384 "
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

# V3 ile aynı fold atamalarını kullandığımızı doğrula.
v3_fold_path = (
    BASE
    / "cross_validation_master408"
    / "cnn_v3_stronger_master408"
    / "fold_assignments_408.csv"
)

if v3_fold_path.exists():
    try:
        v3_fold_df = pd.read_csv(v3_fold_path)

        cmp_fold = df[["id", "oof_fold"]].merge(
            v3_fold_df[["id", "oof_fold"]],
            on="id",
            how="inner",
            suffixes=("_v5", "_v3")
        )

        same_folds = (
            len(cmp_fold) == 408
            and
            (cmp_fold["oof_fold_v5"] == cmp_fold["oof_fold_v3"]).all()
        )

        print(
            "\nV3 ile fold karşılaştırması:",
            "AYNI ✓" if same_folds else "FARKLI ⚠"
        )

        if not same_folds:
            raise RuntimeError(
                "V5 fold atamaları V3 ile aynı değil. "
                "Adil karşılaştırma için eğitim durduruldu."
            )

    except Exception as e:
        raise RuntimeError(
            f"V3 fold karşılaştırması başarısız: {e}"
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
        f"CNN-v5 EXTENDED384 | "
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

    model = V5Extended384().to(DEVICE)

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

        # Eski V3 Stronger kodundaki seçim mantığı korunuyor.
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
# OOF THRESHOLD ANALİZİ
# Sadece 408 OOF tahminlerinden seçilir.
# Bağımsız 24 test setine bakılarak threshold seçilmez.
# =========================================================

def find_best_oof_threshold(pred_df):
    y_true = np.array(
        [LABEL_MAP[x] for x in pred_df["true_label"]],
        dtype=int
    )

    probs = (
        pred_df["fake_probability"]
        .astype(float)
        .to_numpy()
    )

    rows = []

    for threshold in np.arange(
        0.20,
        0.801,
        0.005
    ):
        pred = (
            probs >= threshold
        ).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, pred),
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
        })

    table = pd.DataFrame(rows)

    table["distance_to_050"] = (
        table["threshold"] - 0.50
    ).abs()

    best = (
        table.sort_values(
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

    return float(best["threshold"]), table


OPT_THRESHOLD_TTA, threshold_table_tta = (
    find_best_oof_threshold(oof_tta)
)

OPT_THRESHOLD_NO_TTA, threshold_table_no_tta = (
    find_best_oof_threshold(oof_no_tta)
)

threshold_table_tta.to_csv(
    OUT_DIR / "OOF_threshold_search_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)

threshold_table_no_tta.to_csv(
    OUT_DIR / "OOF_threshold_search_NO_TTA.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\\nOOF optimal threshold | "
    f"TTA={OPT_THRESHOLD_TTA:.3f} | "
    f"NO-TTA={OPT_THRESHOLD_NO_TTA:.3f}"
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
    "CNN-v5 Extended384 | MASTER 408 | Group-Aware 5-Fold OOF",
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
    "CNN-v5 EXTENDED384 | "
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
# V5 EK RAPORLAMA / SUNUM GRAFİKLERİ
# =====================================================================

def _bool_series(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
    )


def save_overall_oof_graphs():
    graph_dir = OUT_DIR / "presentation_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1) Fold TTA / NO-TTA
    # ---------------------------------------------------------
    x = np.arange(len(results))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))

    b1 = ax.bar(
        x - width/2,
        100 * results["accuracy_TTA"].to_numpy(dtype=float),
        width=width,
        label="TTA"
    )

    b2 = ax.bar(
        x + width/2,
        100 * results["accuracy_NO_TTA"].to_numpy(dtype=float),
        width=width,
        label="NO-TTA"
    )

    ax.set_xticks(
        x,
        [f"Fold {i}" for i in range(1, 6)]
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("CNN-v5 Extended384 — 5-Fold OOF Sonuçları")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "01_v5_fold_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---------------------------------------------------------
    # 2) OOF TTA confusion
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_tta)

    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(
        f"V5 OOF TTA Confusion Matrix\n"
        f"Accuracy %{general_tta['accuracy']*100:.2f}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm_tta[i, j]),
                ha="center",
                va="center",
                fontsize=18
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        graph_dir / "02_v5_oof_tta_confusion.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---------------------------------------------------------
    # 3) REAL / FAKE başarı
    # ---------------------------------------------------------
    labels = ["REAL", "FAKE"]
    vals = [
        100 * gc_tta["real_correct"] / gc_tta["real_total"],
        100 * gc_tta["fake_correct"] / gc_tta["fake_total"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    bars = ax.bar(labels, vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doğru sınıflandırma (%)")
    ax.set_title("CNN-v5 OOF TTA — Sınıf Bazlı Başarı")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "03_v5_real_fake_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---------------------------------------------------------
    # 4) Paired / unpaired
    # ---------------------------------------------------------
    pair_rows = (
        subgroups[
            subgroups["group_type"] == "pair_status"
        ]
        .copy()
    )

    if len(pair_rows):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            pair_rows["group_value"],
            100 * pair_rows["accuracy"].to_numpy(dtype=float)
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("NO-TTA doğruluğu (%)")
        ax.set_title("CNN-v5 — Paired / Unpaired Sonuçları")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)

        plt.tight_layout()
        plt.savefig(
            graph_dir / "04_v5_paired_unpaired.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # ---------------------------------------------------------
    # 5) Format
    # ---------------------------------------------------------
    fmt_rows = (
        subgroups[
            subgroups["group_type"] == "format"
        ]
        .copy()
    )

    if len(fmt_rows):
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            fmt_rows["group_value"],
            100 * fmt_rows["accuracy"].to_numpy(dtype=float)
        )
        ax.set_ylim(0, 100)
        ax.set_ylabel("NO-TTA doğruluğu (%)")
        ax.set_title("CNN-v5 — Format Bazlı Sonuç")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)

        plt.tight_layout()
        plt.savefig(
            graph_dir / "05_v5_format_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def compare_v3_v4_v5():
    graph_dir = OUT_DIR / "presentation_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        (
            "V3 Stronger",
            BASE
            / "cross_validation_master408"
            / "cnn_v3_stronger_master408"
        ),
        (
            "V4 MultiScale512",
            BASE
            / "cross_validation_master408"
            / "cnn_v4_multiscale512"
        ),
        (
            "V5 Extended384",
            OUT_DIR
        ),
    ]

    rows = []

    for model_name, folder in candidates:
        tta_path = folder / "OOF_ALL_408_TTA.csv"
        no_path = folder / "OOF_ALL_408_NO_TTA.csv"

        if not tta_path.exists() or not no_path.exists():
            continue

        tta_df = pd.read_csv(tta_path)
        no_df = pd.read_csv(no_path)

        rows.append({
            "model": model_name,
            "tta_accuracy": float(
                _bool_series(tta_df["correct"]).mean()
            ),
            "no_tta_accuracy": float(
                _bool_series(no_df["correct"]).mean()
            ),
        })

    if not rows:
        return

    cmp_df = pd.DataFrame(rows)

    cmp_df.to_csv(
        OUT_DIR / "V3_V4_V5_OOF_COMPARISON.csv",
        index=False,
        encoding="utf-8-sig"
    )

    x = np.arange(len(cmp_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))

    b1 = ax.bar(
        x - width/2,
        100 * cmp_df["tta_accuracy"],
        width=width,
        label="TTA"
    )

    b2 = ax.bar(
        x + width/2,
        100 * cmp_df["no_tta_accuracy"],
        width=width,
        label="NO-TTA"
    )

    ax.set_xticks(x, cmp_df["model"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("408 OOF doğruluğu (%)")
    ax.set_title("V3 vs V4 vs V5 — Aynı 408 / Aynı 5-Fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "06_v3_v4_v5_oof_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =====================================================================
# HUMAN TEST — 24 KATILIMCI vs V5 OOF
# =====================================================================

def human_pairwise_score(model_human_df):
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

        g_prob = float(g["fake_probability"])
        y_prob = float(y["fake_probability"])

        rows.append({
            "pair_no": pair_no,
            "real_code": g_code,
            "fake_code": y_code,
            "real_fake_probability": g_prob,
            "fake_fake_probability": y_prob,
            "margin_fake_minus_real": y_prob - g_prob,
            "correct": y_prob > g_prob
        })

    out = pd.DataFrame(rows)
    return out, int(out["correct"].sum())


def run_human_test_comparison():
    human_path = OUT_DIR / "HUMAN_TEST_24_MODEL_OOF_TTA.csv"

    if not human_path.exists():
        print(
            "\\nHuman karşılaştırması atlandı: "
            "HUMAN_TEST_24_MODEL_OOF_TTA.csv yok."
        )
        return

    h = pd.read_csv(human_path)

    # 24 katılımcının daha önce hesaplanan gerçek Google Form sonuçları
    HUMAN_PAIR_ACC = 58.3
    HUMAN_SINGLE_ACC = 54.5
    HUMAN_OVERALL18_ACC = 55.8

    pair_df, pair_correct = human_pairwise_score(h)

    single = h[
        h["form_section"]
        .astype(str)
        .str.upper()
        ==
        "TEKIL"
    ].copy()

    y_true_single = (
        single["true_label"]
        .map(LABEL_MAP)
        .to_numpy(dtype=int)
    )

    probs_single = (
        single["fake_probability"]
        .astype(float)
        .to_numpy()
    )

    pred_single = (
        probs_single >= 0.50
    ).astype(int)

    single_correct = int(
        (pred_single == y_true_single).sum()
    )

    overall_correct = pair_correct + single_correct

    # 24 görüntüyü tek tek
    y_true_24 = (
        h["true_label"]
        .map(LABEL_MAP)
        .to_numpy(dtype=int)
    )

    probs_24 = (
        h["fake_probability"]
        .astype(float)
        .to_numpy()
    )

    pred_24 = (
        probs_24 >= 0.50
    ).astype(int)

    independent24_correct = int(
        (pred_24 == y_true_24).sum()
    )

    summary = pd.DataFrame([
        {
            "measurement": "Pairwise 6 karar",
            "human_accuracy_pct": HUMAN_PAIR_ACC,
            "v5_accuracy_pct": 100 * pair_correct / 6
        },
        {
            "measurement": "Tekil 12 karar",
            "human_accuracy_pct": HUMAN_SINGLE_ACC,
            "v5_accuracy_pct": 100 * single_correct / 12
        },
        {
            "measurement": "Aynı 18 karar",
            "human_accuracy_pct": HUMAN_OVERALL18_ACC,
            "v5_accuracy_pct": 100 * overall_correct / 18
        },
    ])

    summary.to_csv(
        OUT_DIR / "HUMAN_VS_V5_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pair_df.to_csv(
        OUT_DIR / "HUMAN_V5_PAIRWISE_6_DETAIL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    graph_dir = OUT_DIR / "presentation_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(3)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))

    b1 = ax.bar(
        x - width/2,
        summary["human_accuracy_pct"],
        width=width,
        label="İnsan ortalaması"
    )

    b2 = ax.bar(
        x + width/2,
        summary["v5_accuracy_pct"],
        width=width,
        label="CNN-v5"
    )

    ax.set_xticks(
        x,
        ["Eşli 6", "Tekil 12", "Genel 18"]
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("İnsan vs CNN-v5 — Aynı Görev")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(b1, fmt="%.1f%%", padding=3)
    ax.bar_label(b2, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "07_human_vs_v5.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(
        f"\\nHUMAN TEST V5 | "
        f"18 karar: {overall_correct}/18 "
        f"(%{100*overall_correct/18:.1f}) | "
        f"24 görüntü tek tek: {independent24_correct}/24 "
        f"(%{100*independent24_correct/24:.1f})"
    )


# =====================================================================
# BAĞIMSIZ 24 DIŞ TEST — V5 5-FOLD ENSEMBLE
# =====================================================================

TEST_CSV = BASE / "test" / "test_dataset_pairli.csv"


def resolve_external_path(raw_path):
    raw = str(raw_path).strip()

    direct = Path(raw)
    if direct.exists():
        return direct.resolve()

    wp = PureWindowsPath(raw)

    candidate = BASE / "test" / wp.name
    if candidate.exists():
        return candidate.resolve()

    matches = [
        p
        for p in (BASE / "test").rglob(wp.name)
        if p.is_file()
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Bağımsız test görüntüsü bulunamadı: {raw}"
    )


@torch.no_grad()
def predict_tensor(model, tensor, tta=True):
    model.eval()

    probs = []

    for start in range(0, len(tensor), BATCH):
        x = tensor[start:start+BATCH]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits_normal = model(x)

            if tta:
                logits_flip = model(
                    torch.flip(x, dims=[3])
                )

                logits = (
                    logits_normal
                    +
                    logits_flip
                ) / 2.0
            else:
                logits = logits_normal

        probs.extend(
            torch.sigmoid(
                logits.float()
            )
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    return np.asarray(probs, dtype=float)


def external_metrics(y_true, probs, threshold=0.50):
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
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probs
        ),
        "real_correct": int(tn),
        "real_total": int(tn + fp),
        "fake_correct": int(tp),
        "fake_total": int(tp + fn),
        "cm": cm,
        "pred": pred,
    }


def run_external_24():
    if not TEST_CSV.exists():
        print(
            "\\nBağımsız 24 test atlandı: "
            "test/test_dataset_pairli.csv bulunamadı."
        )
        return None

    ext = pd.read_csv(TEST_CSV)

    ext["label"] = (
        ext["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    ext["pair_no"] = pd.to_numeric(
        ext["no"],
        errors="raise"
    ).astype(int)

    ext["resolved_path"] = [
        str(resolve_external_path(p))
        for p in ext["file_path"]
    ]

    if len(ext) != 24:
        raise RuntimeError(
            f"Bağımsız test 24 görüntü olmalı. Bulunan={len(ext)}"
        )

    if (ext["label"] == "REAL").sum() != 12:
        raise RuntimeError("Bağımsız test REAL sayısı 12 değil.")

    if (ext["label"] == "FAKE").sum() != 12:
        raise RuntimeError("Bağımsız test FAKE sayısı 12 değil.")

    X_ext = torch.stack(
        [
            load_rgb(p)
            for p in ext["resolved_path"]
        ]
    ).to(DEVICE)

    y_true = (
        ext["label"]
        .map(LABEL_MAP)
        .to_numpy(dtype=int)
    )

    fold_probs_tta = []
    fold_probs_no = []
    fold_rows = []

    print("\\nBağımsız 24 test — V5:")

    for fold in range(1, 6):
        fold_dir = OUT_DIR / f"fold_{fold}"

        ckpts = list(
            fold_dir.glob("BEST_MODEL_epoch*.pth")
        )

        if len(ckpts) != 1:
            raise RuntimeError(
                f"Fold {fold}: tek BEST_MODEL bekleniyor."
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

        model = V5Extended384().to(DEVICE)
        model.load_state_dict(state)

        p_tta = predict_tensor(
            model,
            X_ext,
            tta=True
        )

        p_no = predict_tensor(
            model,
            X_ext,
            tta=False
        )

        fold_probs_tta.append(p_tta)
        fold_probs_no.append(p_no)

        mt = external_metrics(
            y_true,
            p_tta,
            0.50
        )

        mn = external_metrics(
            y_true,
            p_no,
            0.50
        )

        fold_rows.append({
            "fold": fold,
            "tta_accuracy": mt["accuracy"],
            "no_tta_accuracy": mn["accuracy"],
            "tta_real_correct": mt["real_correct"],
            "tta_fake_correct": mt["fake_correct"],
        })

        print(
            f"  Fold {fold}: "
            f"TTA %{100*mt['accuracy']:.2f} | "
            f"NO-TTA %{100*mn['accuracy']:.2f}"
        )

        del model
        torch.cuda.empty_cache()

    P_TTA = np.stack(fold_probs_tta, axis=0)
    P_NO = np.stack(fold_probs_no, axis=0)

    ensemble_tta = P_TTA.mean(axis=0)
    ensemble_no = P_NO.mean(axis=0)

    m_tta_050 = external_metrics(
        y_true,
        ensemble_tta,
        0.50
    )

    m_no_050 = external_metrics(
        y_true,
        ensemble_no,
        0.50
    )

    m_tta_oof = external_metrics(
        y_true,
        ensemble_tta,
        OPT_THRESHOLD_TTA
    )

    ext["v5_fake_probability_tta"] = ensemble_tta
    ext["v5_fake_probability_no_tta"] = ensemble_no
    ext["v5_pred_tta_050"] = [
        LABEL_NAME[x]
        for x in m_tta_050["pred"]
    ]

    # Pairwise
    pair_rows = []

    for pair_no, g in ext.groupby(
        "pair_no",
        sort=True
    ):
        real = g[
            g["label"] == "REAL"
        ].iloc[0]

        fake = g[
            g["label"] == "FAKE"
        ].iloc[0]

        rprob = float(
            real["v5_fake_probability_tta"]
        )

        fprob = float(
            fake["v5_fake_probability_tta"]
        )

        pair_rows.append({
            "pair_no": int(pair_no),
            "real_probability": rprob,
            "fake_probability": fprob,
            "margin": fprob - rprob,
            "correct": fprob > rprob
        })

    pair_df = pd.DataFrame(pair_rows)
    pair_correct = int(pair_df["correct"].sum())

    ext_dir = OUT_DIR / "independent_external_test_24"
    ext_dir.mkdir(parents=True, exist_ok=True)

    ext.to_csv(
        ext_dir / "independent24_v5_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(fold_rows).to_csv(
        ext_dir / "fold_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pair_df.to_csv(
        ext_dir / "pairwise_12_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame([
        {
            "mode": "TTA_0.50",
            **{
                k: v
                for k, v in m_tta_050.items()
                if k not in {"cm", "pred"}
            }
        },
        {
            "mode": f"TTA_OOF_THRESHOLD_{OPT_THRESHOLD_TTA:.3f}",
            **{
                k: v
                for k, v in m_tta_oof.items()
                if k not in {"cm", "pred"}
            }
        },
        {
            "mode": "NO_TTA_0.50",
            **{
                k: v
                for k, v in m_no_050.items()
                if k not in {"cm", "pred"}
            }
        },
    ]).to_csv(
        ext_dir / "independent24_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    graph_dir = OUT_DIR / "presentation_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # External confusion
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(m_tta_050["cm"])
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(
        "V5 — Bağımsız 24 Test / 5-Fold Ensemble TTA\n"
        f"Accuracy %{100*m_tta_050['accuracy']:.2f}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(m_tta_050["cm"][i, j]),
                ha="center",
                va="center",
                fontsize=18
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        graph_dir / "08_v5_external24_confusion.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Individual vs pairwise
    vals = [
        100 * m_tta_050["accuracy"],
        100 * pair_correct / 12
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        ["24 görüntü\ntek tek", "12 çift\npairwise"],
        vals
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("V5 — Bağımsız Dış Test")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "09_v5_external_individual_vs_pairwise.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(
        f"\\nV5 BAĞIMSIZ 24 ENSEMBLE TTA: "
        f"{m_tta_050['real_correct'] + m_tta_050['fake_correct']}/24 "
        f"(%{100*m_tta_050['accuracy']:.2f}) | "
        f"REAL {m_tta_050['real_correct']}/12 | "
        f"FAKE {m_tta_050['fake_correct']}/12 | "
        f"PAIR {pair_correct}/12"
    )

    del X_ext
    torch.cuda.empty_cache()

    return ext


# =====================================================================
# SENTETİK / FAKE GÖRÜNTÜLERE NOISE TESTİ
# Eğitim sırasında noise kullanılmaz.
# Bağımsız 24 setindeki 12 FAKE görüntü test edilir.
# =====================================================================

NOISE_LEVELS = [
    ("ORIGINAL", 0.00),
    ("HAFIF", 0.02),
    ("ORTA", 0.05),
]

NOISE_SEED = 2026


def build_noise_versions(clean_cpu):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(NOISE_SEED)

    base_noise = torch.randn(
        clean_cpu.shape,
        generator=generator,
        dtype=clean_cpu.dtype
    )

    batches = []

    for level_name, sigma in NOISE_LEVELS:
        if sigma == 0:
            noisy = clean_cpu.clone()
        else:
            noisy = torch.clamp(
                clean_cpu + sigma * base_noise,
                0.0,
                1.0
            )

        batches.append(noisy)

    return torch.cat(batches, dim=0)


def run_v5_noise_test():
    if not TEST_CSV.exists():
        print(
            "\\nNoise testi atlandı: bağımsız test CSV yok."
        )
        return

    ext = pd.read_csv(TEST_CSV)

    ext["label"] = (
        ext["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    fake = (
        ext[
            ext["label"] == "FAKE"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(fake) != 12:
        raise RuntimeError(
            f"Noise testi için 12 FAKE bekleniyor. Bulunan={len(fake)}"
        )

    fake["resolved_path"] = [
        str(resolve_external_path(p))
        for p in fake["file_path"]
    ]

    clean_cpu = torch.stack(
        [
            load_rgb(p)
            for p in fake["resolved_path"]
        ]
    )

    all_noise_cpu = build_noise_versions(
        clean_cpu
    )

    all_noise = all_noise_cpu.to(DEVICE)

    fold_probs = []

    for fold in range(1, 6):
        fold_dir = OUT_DIR / f"fold_{fold}"

        ckpts = list(
            fold_dir.glob("BEST_MODEL_epoch*.pth")
        )

        if len(ckpts) != 1:
            raise RuntimeError(
                f"Noise testi Fold {fold}: checkpoint yok."
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

        model = V5Extended384().to(DEVICE)
        model.load_state_dict(state)

        fold_probs.append(
            predict_tensor(
                model,
                all_noise,
                tta=True
            )
        )

        del model
        torch.cuda.empty_cache()

    ensemble = np.stack(
        fold_probs,
        axis=0
    ).mean(axis=0)

    rows = []
    n = 12

    for level_idx, (
        level_name,
        sigma
    ) in enumerate(NOISE_LEVELS):

        start = level_idx * n
        end = start + n

        probs = ensemble[start:end]
        pred_fake = probs >= 0.50

        for i in range(n):
            rows.append({
                "noise_level": level_name,
                "sigma": sigma,
                "image_index": i,
                "file_name": fake.loc[i, "file_name"],
                "format": fake.loc[i, "format"],
                "generator": fake.loc[i, "generator"],
                "fake_probability": float(probs[i]),
                "predicted_label": (
                    "FAKE" if pred_fake[i] else "REAL"
                ),
                "correct_fake_detection": bool(pred_fake[i])
            })

    detail = pd.DataFrame(rows)

    summary_rows = []

    for level_name, sigma in NOISE_LEVELS:
        g = detail[
            detail["noise_level"] == level_name
        ]

        summary_rows.append({
            "noise_level": level_name,
            "sigma": sigma,
            "fake_correct": int(
                g["correct_fake_detection"].sum()
            ),
            "fake_total": len(g),
            "fake_detection_accuracy": float(
                g["correct_fake_detection"].mean()
            ),
            "mean_fake_probability": float(
                g["fake_probability"].mean()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    noise_dir = OUT_DIR / "synthetic_noise_test"
    noise_dir.mkdir(parents=True, exist_ok=True)

    detail.to_csv(
        noise_dir / "noise_test_v5_detail.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        noise_dir / "noise_test_v5_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    graph_dir = OUT_DIR / "presentation_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Fake detection success
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        ["Orijinal", "Hafif Noise", "Orta Noise"],
        100 * summary["fake_detection_accuracy"]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("FAKE doğru tespit (%)")
    ax.set_title(
        "CNN-v5 — Sentetik Görüntülerde Gürültü Testi"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "10_v5_noise_fake_detection.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Mean fake probability
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        ["Orijinal", "Hafif Noise", "Orta Noise"],
        100 * summary["mean_fake_probability"],
        marker="o",
        linewidth=2
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1.3,
        label="FAKE eşiği %50"
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Ortalama FAKE olasılığı (%)")
    ax.set_title(
        "CNN-v5 — Noise Arttıkça FAKE Güveni"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        graph_dir / "11_v5_noise_mean_fake_probability.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\\nV5 SENTETİK NOISE TESTİ:")

    for _, r in summary.iterrows():
        print(
            f"  {r['noise_level']:8s} | "
            f"FAKE {int(r['fake_correct'])}/{int(r['fake_total'])} "
            f"(%{100*r['fake_detection_accuracy']:.2f}) | "
            f"Ort.P(FAKE) %{100*r['mean_fake_probability']:.2f}"
        )

    del all_noise
    torch.cuda.empty_cache()


# =====================================================================
# FULL PIPELINE SONU
# =====================================================================

save_overall_oof_graphs()
compare_v3_v4_v5()
run_human_test_comparison()
run_external_24()
run_v5_noise_test()

print(
    "\\n"
    +
    "=" * 88
)

print(
    "CNN-v5 EXTENDED384 FULL PIPELINE TAMAMLANDI"
)

print(
    "=" * 88
)

print(
    "\\nMİMARİ:"
)

print(
    "  V3 gövdesi: 64→128→256 Residual+ECA"
)

print(
    "  Yeni stage: 256→384 + Residual+ECA"
)

print(
    "  Pooling: AvgPool2×2 + MaxPool2×2"
)

print(
    "  Classifier: 3072→256→64→1"
)

print(
    f"  Label smoothing: {LABEL_SMOOTH}"
)

print(
    f"  Max epoch / patience: {MAX_EPOCHS}/{PATIENCE}"
)

print(
    f"  LR: {LR} | Min LR: {MIN_LR}"
)

print(
    f"\\nOOF TTA Accuracy: %{100*general_tta['accuracy']:.2f}"
)

print(
    f"OOF NO-TTA Accuracy: %{100*general_no_tta['accuracy']:.2f}"
)

print(
    f"OOF optimal TTA threshold: {OPT_THRESHOLD_TTA:.3f}"
)

print(
    "\\nSonuç klasörü:"
)

print(
    OUT_DIR
)
