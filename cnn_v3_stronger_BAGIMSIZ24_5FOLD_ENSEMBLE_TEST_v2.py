# -*- coding: utf-8 -*-
"""
CNN-v3 STRONGER — 408'de eğitilen 5 fold model ile
24 GÖRÜNTÜLÜK BAĞIMSIZ TEST SETİ / 5-FOLD ENSEMBLE

Amaç:
- test/test_dataset_pairli.csv içindeki 12 REAL + 12 FAKE = 24 bağımsız görüntüyü test eder.
- 408 görüntü üzerinde eğitilmiş 5 fold'un BEST_MODEL checkpoint'lerini kullanır.
- Her fold'u tek tek raporlar.
- 5 fold olasılıklarının ortalamasını alıp ENSEMBLE sonucu üretir.
- TTA ve NO-TTA ayrı hesaplanır.
- 24 görüntüyü tek tek REAL/FAKE sınıflandırır.
- 12 eşli çift için ayrıca "hangisi daha yapay?" pairwise testini hesaplar.
- Confusion matrix, fold karşılaştırması, sınıf başarısı, çift marjları,
  görüntü bazlı olasılıklar, belirsizlik ve hata görselleri üretir.
- Format / generator bilgisi CSV'de varsa veya path'ten çıkarılabiliyorsa
  bunlara göre alt-grup analizi de üretir.
- 408 OOF dosyaları bulunursa 408-CV ile bağımsız test sonucu aynı grafikte gösterilir.

ÖNEMLİ METODOLOJİ:
Bu 24 görüntü 408 eğitim/CV havuzuna dahil değildir.
Bu script eğitim YAPMAZ; yalnızca daha önce eğitilmiş 5 fold modeli test eder.

Beklenen proje yapısı:

C:\\Users\\Ceren\\Downloads\\yapay sinir ağları\\
│
├─ cross_validation_master408\\
│  └─ cnn_v3_stronger_master408\\
│     ├─ fold_1\\BEST_MODEL_epoch....pth
│     ├─ fold_2\\BEST_MODEL_epoch....pth
│     ├─ fold_3\\BEST_MODEL_epoch....pth
│     ├─ fold_4\\BEST_MODEL_epoch....pth
│     └─ fold_5\\BEST_MODEL_epoch....pth
│
├─ test\\
│  ├─ test_dataset_pairli.csv
│  └─ görüntü klasörleri...
│
└─ cnn_v3_stronger_BAGIMSIZ24_5FOLD_ENSEMBLE_TEST.py

Çıktı:
bagimsiz_test_24_v3_stronger_5fold_ensemble/
"""

from pathlib import Path, PureWindowsPath
import io
import re
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# =========================================================
# AYARLAR
# =========================================================

BASE = Path(__file__).resolve().parent
TEST_DIR = BASE / "test"
TEST_CSV = TEST_DIR / "test_dataset_pairli.csv"

MODEL_DIR = (
    BASE
    / "cross_validation_master408"
    / "cnn_v3_stronger_master408"
)

OUT_DIR = (
    BASE
    / "bagimsiz_test_24_v3_stronger_5fold_ensemble"
)

SIZE = 320
BATCH = 24

LABEL_MAP = {"REAL": 0, "FAKE": 1}
LABEL_NAME = {0: "REAL", 1: "FAKE"}

HEIC_EXTS = {".heic", ".heif"}

OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11


# =========================================================
# İSİM NORMALİZASYONU / CSV OTOMATİK TANIMA
# =========================================================

TR_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})


def norm(s):
    s = str(s).translate(TR_MAP).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def find_col(columns, must_any_groups, prefer=None):
    """
    must_any_groups:
      [["real","gercek"], ["path","yol"]]
    Her grup içinden en az bir token sütun adında bulunmalı.
    """
    scored = []

    for c in columns:
        n = norm(c)

        ok = True
        for group in must_any_groups:
            if not any(token in n for token in group):
                ok = False
                break

        if ok:
            score = 0

            if prefer:
                for token in prefer:
                    if token in n:
                        score += 1

            scored.append((score, c))

    if not scored:
        return None

    scored.sort(
        key=lambda x: (x[0], -len(norm(x[1]))),
        reverse=True,
    )

    return scored[0][1]


# =========================================================
# PATH ÇÖZÜMLEME
# =========================================================

def resolve_test_path(raw_path):
    raw = str(raw_path).strip().strip('"')

    p = Path(raw)

    if p.exists():
        return p.resolve()

    # Relative path ise test altından dene
    p2 = TEST_DIR / raw

    if p2.exists():
        return p2.resolve()

    # Windows absolute path bozulduysa sadece son dosya adından bul
    wp = PureWindowsPath(raw)
    name = wp.name

    # test klasöründe exact filename ara
    matches = [
        p for p in TEST_DIR.rglob(name)
        if p.is_file()
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    # Büyük/küçük harf farkına karşı
    lower_name = name.casefold()

    matches = [
        p for p in TEST_DIR.rglob("*")
        if p.is_file()
        and p.name.casefold() == lower_name
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            f"Aynı isimde birden fazla test dosyası bulundu:\n"
            f"{name}\n"
            + "\n".join(str(x) for x in matches)
        )

    raise FileNotFoundError(
        f"Test görüntüsü bulunamadı:\n{raw}"
    )


# =========================================================
# META BİLGİ ÇIKARIMI
# =========================================================

def infer_format(path, fallback=""):
    if fallback and str(fallback).strip():
        x = str(fallback).upper().strip()

        if x in {"JPG", "JPEG"}:
            return "JPEG"

        if x in {"HEIC", "HEIF"}:
            return "HEIC"

        if x == "PNG":
            return "PNG"

    ext = Path(path).suffix.lower()

    if ext in {".jpg", ".jpeg"}:
        return "JPEG"

    if ext in {".heic", ".heif"}:
        return "HEIC"

    if ext == ".png":
        return "PNG"

    return ext.replace(".", "").upper() or "UNKNOWN"


def infer_generator(path, label, fallback=""):
    if fallback and str(fallback).strip() and str(fallback).lower() != "nan":
        x = str(fallback).strip()

        if label == "REAL":
            return "NONE"

        return x

    if label == "REAL":
        return "NONE"

    s = str(path).lower()

    if "gemini" in s:
        return "Gemini"

    if (
        "chat" in s
        or "gbt" in s
        or "gpt" in s
    ):
        return "ChatGPT"

    return "UNKNOWN"


# =========================================================
# TEST CSV'Yİ 24 SATIRLIK LONG FORMATA ÇEVİR
# =========================================================

def load_test_csv():
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV bulunamadı:\n{TEST_CSV}"
        )

    raw = pd.read_csv(TEST_CSV)

    print("\nTest CSV sütunları:")
    for c in raw.columns:
        print("  -", c)

    cols = list(raw.columns)

    # -----------------------------------------------------
    # 1) WIDE / PAIR FORMAT:
    # Her satırda REAL ve FAKE birlikte.
    # -----------------------------------------------------

    real_path_col = find_col(
        cols,
        [["real", "gercek"], ["path", "yol"]],
        prefer=["dosya"],
    )

    fake_path_col = find_col(
        cols,
        [["fake", "yapay"], ["path", "yol"]],
        prefer=["dosya"],
    )

    if real_path_col and fake_path_col:
        print("\nCSV biçimi: 12 satırlık EŞLİ/WIDE format algılandı.")
        print("REAL path sütunu:", real_path_col)
        print("FAKE path sütunu:", fake_path_col)

        pair_col = (
            find_col(cols, [["pair", "cift"]])
            or find_col(cols, [["no"]])
        )

        real_name_col = find_col(
            cols,
            [["real", "gercek"], ["ad", "name", "isim"]],
        )

        fake_name_col = find_col(
            cols,
            [["fake", "yapay"], ["ad", "name", "isim"]],
        )

        real_format_col = find_col(
            cols,
            [["real", "gercek"], ["format"]],
        )

        fake_format_col = find_col(
            cols,
            [["fake", "yapay"], ["format"]],
        )

        fake_gen_col = find_col(
            cols,
            [["fake", "yapay"], ["uretic", "generator"]],
        )

        rows = []

        for i, r in raw.iterrows():
            pair_no = (
                r[pair_col]
                if pair_col is not None
                else i + 1
            )

            rp = resolve_test_path(
                r[real_path_col]
            )

            fp = resolve_test_path(
                r[fake_path_col]
            )

            rows.append({
                "pair_no": int(pair_no),
                "pair_id": f"test_pair_{int(pair_no):02d}",
                "within_pair": "REAL",
                "label": "REAL",
                "file_name": (
                    str(r[real_name_col])
                    if real_name_col is not None
                    else rp.name
                ),
                "file_path": str(rp),
                "format": infer_format(
                    rp,
                    r[real_format_col]
                    if real_format_col is not None
                    else "",
                ),
                "generator": "NONE",
            })

            rows.append({
                "pair_no": int(pair_no),
                "pair_id": f"test_pair_{int(pair_no):02d}",
                "within_pair": "FAKE",
                "label": "FAKE",
                "file_name": (
                    str(r[fake_name_col])
                    if fake_name_col is not None
                    else fp.name
                ),
                "file_path": str(fp),
                "format": infer_format(
                    fp,
                    r[fake_format_col]
                    if fake_format_col is not None
                    else "",
                ),
                "generator": infer_generator(
                    fp,
                    "FAKE",
                    r[fake_gen_col]
                    if fake_gen_col is not None
                    else "",
                ),
            })

        df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # 2) LONG FORMAT:
    # 24 satır, her satır bir görüntü.
    # -----------------------------------------------------

    else:
        print("\nCSV biçimi: LONG format deneniyor.")

        path_col = find_col(
            cols,
            [["path", "yol"]],
        )

        label_col = find_col(
            cols,
            [["label", "etiket", "sinif"]],
        )

        if path_col is None or label_col is None:
            raise RuntimeError(
                "CSV otomatik okunamadı.\n"
                "Beklenen iki biçimden biri:\n"
                "1) 12 satır: REAL Dosya Yolu + FAKE Dosya Yolu\n"
                "2) 24 satır: file_path + label + pair_no/group_id"
            )

        pair_col = (
            find_col(cols, [["pair", "cift"]])
            or find_col(cols, [["group"]])
        )

        # Bu test CSV'sinde eşleşme bilgisi doğrudan "no" sütununda.
        # Aynı no değeri bir REAL ve bir FAKE satırında tekrar ediyor.
        # Örn: no=1 -> 1 REAL + 1 FAKE, no=2 -> 1 REAL + 1 FAKE ...
        if pair_col is None:
            exact_no_cols = [
                c for c in cols
                if norm(c) == "no"
            ]

            if exact_no_cols:
                candidate_no = exact_no_cols[0]

                # "no" gerçekten 12 eşli çift kimliği mi kontrol et.
                no_series = raw[candidate_no]

                value_counts = (
                    no_series
                    .astype(str)
                    .str.strip()
                    .value_counts()
                )

                if (
                    len(value_counts) == 12
                    and
                    (value_counts == 2).all()
                ):
                    pair_col = candidate_no
                    print(
                        f'Eşli çift bilgisi "{pair_col}" sütunundan alındı '
                        '(12 değer, her biri 2 kez).'
                    )

        name_col = find_col(
            cols,
            [["file", "dosya"], ["name", "ad", "isim"]],
        )

        format_col = find_col(
            cols,
            [["format"]],
        )

        gen_col = find_col(
            cols,
            [["generator", "uretic"]],
        )

        temp = []

        for i, r in raw.iterrows():
            label_raw = norm(r[label_col])

            if (
                "real" in label_raw
                or "gercek" in label_raw
            ):
                label = "REAL"

            elif (
                "fake" in label_raw
                or "yapay" in label_raw
            ):
                label = "FAKE"

            else:
                raise RuntimeError(
                    f"Bilinmeyen etiket: {r[label_col]}"
                )

            p = resolve_test_path(
                r[path_col]
            )

            temp.append({
                "pair_no_raw": (
                    r[pair_col]
                    if pair_col is not None
                    else np.nan
                ),
                "label": label,
                "file_name": (
                    str(r[name_col])
                    if name_col is not None
                    else p.name
                ),
                "file_path": str(p),
                "format": infer_format(
                    p,
                    r[format_col]
                    if format_col is not None
                    else "",
                ),
                "generator": infer_generator(
                    p,
                    label,
                    r[gen_col]
                    if gen_col is not None
                    else "",
                ),
            })

        df = pd.DataFrame(temp)

        if pair_col is None:
            raise RuntimeError(
                "LONG formatta pair/cift/group sütunu bulunamadı. "
                "12 eşli çiftin hangi görüntüler olduğunu belirlemek için "
                "pair bilgisi gerekli."
            )

        # group/pair değerlerini 1..12'ye map et
        unique_pairs = list(
            dict.fromkeys(
                df["pair_no_raw"].astype(str).tolist()
            )
        )

        pair_map = {
            p: i + 1
            for i, p in enumerate(unique_pairs)
        }

        df["pair_no"] = (
            df["pair_no_raw"]
            .astype(str)
            .map(pair_map)
        )

        df["pair_id"] = df["pair_no"].map(
            lambda x: f"test_pair_{int(x):02d}"
        )

        df["within_pair"] = df["label"]

        df = df.drop(
            columns=["pair_no_raw"]
        )

    # -----------------------------------------------------
    # DOĞRULAMA
    # -----------------------------------------------------

    df = df.sort_values(
        ["pair_no", "label"]
    ).reset_index(drop=True)

    if len(df) != 24:
        raise RuntimeError(
            f"Bağımsız test 24 görüntü olmalı. Bulunan: {len(df)}"
        )

    if (df["label"] == "REAL").sum() != 12:
        raise RuntimeError(
            "REAL sayısı 12 değil."
        )

    if (df["label"] == "FAKE").sum() != 12:
        raise RuntimeError(
            "FAKE sayısı 12 değil."
        )

    pair_sizes = df.groupby("pair_no").size()

    if len(pair_sizes) != 12 or not (pair_sizes == 2).all():
        raise RuntimeError(
            "12 çiftin her birinde tam 2 görüntü olmalı."
        )

    pair_labels = (
        df.groupby("pair_no")["label"]
        .apply(lambda x: set(x))
    )

    bad_pairs = [
        k for k, v in pair_labels.items()
        if v != {"REAL", "FAKE"}
    ]

    if bad_pairs:
        raise RuntimeError(
            f"REAL+FAKE içermeyen çiftler: {bad_pairs}"
        )

    df["true_num"] = df["label"].map(
        LABEL_MAP
    )

    df["test_id"] = np.arange(
        1,
        len(df) + 1
    )

    print(
        "\nBağımsız test doğrulandı:"
    )

    print(
        f"  Toplam: {len(df)}"
    )

    print(
        f"  REAL: {(df['label']=='REAL').sum()}"
    )

    print(
        f"  FAKE: {(df['label']=='FAKE').sum()}"
    )

    print(
        f"  Çift: {df['pair_no'].nunique()}"
    )

    print(
        "\nFormat dağılımı:"
    )

    print(
        df.groupby(["label", "format"])
        .size()
        .to_string()
    )

    print(
        "\nGenerator dağılımı:"
    )

    print(
        df.groupby(["label", "generator"])
        .size()
        .to_string()
    )

    return df


# =========================================================
# HEIC / RGB OKUMA
# =========================================================

def decode_heic_with_imagemagick(path):
    if shutil.which("magick") is None:
        raise RuntimeError(
            "HEIC görüntüleri okumak için ImageMagick gerekli. "
            "PowerShell'de 'magick -version' çalışmalı."
        )

    result = subprocess.run(
        [
            "magick",
            str(path),
            "-auto-orient",
            "png:-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"HEIC açılamadı:\n{path}\n\n"
            + result.stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    with Image.open(
        io.BytesIO(result.stdout)
    ) as img:
        return img.convert("RGB").copy()


def open_original_rgb(path):
    path = Path(path)

    if path.suffix.lower() in HEIC_EXTS:
        return decode_heic_with_imagemagick(path)

    with Image.open(path) as raw:
        return (
            ImageOps.exif_transpose(raw)
            .convert("RGB")
            .copy()
        )


def load_model_tensor(path):
    img = open_original_rgb(path)

    img = ImageOps.fit(
        img,
        (SIZE, SIZE),
        method=Image.Resampling.LANCZOS,
    )

    arr = (
        np.asarray(
            img,
            dtype=np.float32,
        )
        / 255.0
    )

    return torch.from_numpy(
        np.transpose(
            arr,
            (2, 0, 1),
        ).copy()
    )


# =========================================================
# CNN-v3 STRONGER — EĞİTİMDEKİ MİMARİYLE AYNI
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
            bias=False,
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


class ResidualECABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.eca = ECAAttention(
            kernel_size=3
        )

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
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block64 = ResidualECABlock(64)

        self.down128 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block128 = ResidualECABlock(128)

        self.down256 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block256 = ResidualECABlock(256)

        self.avg2 = nn.AdaptiveAvgPool2d(
            (2, 2)
        )

        self.max2 = nn.AdaptiveMaxPool2d(
            (2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(32, 1),
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
            1,
        )

        mx = torch.flatten(
            self.max2(x),
            1,
        )

        x = torch.cat(
            [avg, mx],
            dim=1,
        )

        return self.classifier(x).squeeze(1)


# =========================================================
# CHECKPOINT BUL
# =========================================================

def find_checkpoints():
    checkpoints = []

    for fold in range(1, 6):
        fold_dir = MODEL_DIR / f"fold_{fold}"

        if not fold_dir.exists():
            raise FileNotFoundError(
                f"Fold klasörü bulunamadı:\n{fold_dir}"
            )

        matches = sorted(
            fold_dir.glob(
                "BEST_MODEL_epoch*.pth"
            )
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Fold {fold} için tam 1 BEST_MODEL bekleniyor. "
                f"Bulunan: {len(matches)}\n"
                + "\n".join(
                    str(x)
                    for x in matches
                )
            )

        checkpoints.append(
            (fold, matches[0])
        )

    print(
        "\n5 checkpoint bulundu:"
    )

    for fold, p in checkpoints:
        print(
            f"  Fold {fold}: {p.name}"
        )

    return checkpoints


# =========================================================
# MODEL TAHMİN
# =========================================================

def load_state(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


@torch.no_grad()
def predict_model(
    model,
    X,
    tta,
):
    model.eval()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        logits_normal = model(X)

        if tta:
            X_flip = torch.flip(
                X,
                dims=[3],
            )

            logits_flip = model(
                X_flip
            )

            logits = (
                logits_normal
                +
                logits_flip
            ) / 2.0

        else:
            logits = logits_normal

    prob = torch.sigmoid(
        logits.float()
    )

    return (
        prob.detach()
        .cpu()
        .numpy()
    )


# =========================================================
# METRİK
# =========================================================

def metric_dict(y_true, prob):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    prob = np.asarray(
        prob,
        dtype=float,
    )

    pred = (
        prob >= 0.5
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            pred,
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "recall_fake": recall_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "f1_fake": f1_score(
            y_true,
            pred,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            prob,
        ),
        "real_correct": int(tn),
        "real_total": int(tn + fp),
        "fake_correct": int(tp),
        "fake_total": int(tp + fn),
        "pred": pred,
        "cm": cm,
    }


def pairwise_metric(
    df,
    prob_col,
):
    rows = []

    for pair_no, g in df.groupby(
        "pair_no",
        sort=True,
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

        correct = (
            fake_prob > real_prob
        )

        rows.append({
            "pair_no": int(pair_no),
            "real_file": real["file_name"],
            "fake_file": fake["file_name"],
            "real_fake_probability": real_prob,
            "fake_fake_probability": fake_prob,
            "fake_minus_real_margin": (
                fake_prob - real_prob
            ),
            "pairwise_correct": correct,
        })

    out = pd.DataFrame(rows)

    return out, float(
        out["pairwise_correct"].mean()
    )


# =========================================================
# GRAFİK YARDIMCILARI
# =========================================================

def savefig(name):
    plt.tight_layout()

    plt.savefig(
        OUT_DIR / name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def label_bars(ax, fmt="{:.1f}%"):
    for container in ax.containers:
        try:
            labels = [
                fmt.format(v)
                for v
                in container.datavalues
            ]

            ax.bar_label(
                container,
                labels=labels,
                padding=3,
                fontsize=9,
            )

        except Exception:
            pass


def plot_confusion(
    cm,
    title,
    filename,
):
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(cm)

    ax.set_xticks(
        [0, 1],
        ["REAL", "FAKE"],
    )

    ax.set_yticks(
        [0, 1],
        ["REAL", "FAKE"],
    )

    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18,
            )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    savefig(filename)


# =========================================================
# HATA MONTAJI
# =========================================================

def make_error_montage(
    df,
    prob_col,
    pred_col,
    filename,
    title,
):
    wrong = df[
        df[pred_col] != df["label"]
    ].copy()

    if len(wrong) == 0:
        return

    thumb_w = 280
    thumb_h = 220
    cell_h = 290
    cols = min(3, len(wrong))
    rows = int(
        np.ceil(
            len(wrong) / cols
        )
    )

    canvas = Image.new(
        "RGB",
        (
            cols * thumb_w,
            rows * cell_h + 60,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (10, 10),
        title,
        fill="black",
    )

    for k, (_, r) in enumerate(
        wrong.iterrows()
    ):
        rr = k // cols
        cc = k % cols

        x0 = cc * thumb_w
        y0 = rr * cell_h + 60

        img = open_original_rgb(
            r["file_path"]
        )

        img.thumbnail(
            (
                thumb_w - 20,
                thumb_h - 20,
            ),
            Image.Resampling.LANCZOS,
        )

        bg = Image.new(
            "RGB",
            (
                thumb_w,
                thumb_h,
            ),
            "white",
        )

        ix = (
            thumb_w - img.width
        ) // 2

        iy = (
            thumb_h - img.height
        ) // 2

        bg.paste(
            img,
            (ix, iy),
        )

        canvas.paste(
            bg,
            (x0, y0),
        )

        txt = (
            f"Pair {int(r['pair_no'])} | "
            f"Gerçek: {r['label']} | "
            f"Tahmin: {r[pred_col]}\n"
            f"FAKE olasılığı: %{100*float(r[prob_col]):.1f}"
        )

        draw.text(
            (
                x0 + 5,
                y0 + thumb_h + 5,
            ),
            txt,
            fill="black",
        )

    canvas.save(
        OUT_DIR / filename
    )


# =========================================================
# ANA
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU bulunamadı."
    )

DEVICE = torch.device("cuda")

print(
    "\nGPU:",
    torch.cuda.get_device_name(0),
)

test_df = load_test_csv()
checkpoints = find_checkpoints()

# 24 görüntüyü bir kez oku
print(
    "\n24 test görüntüsü yükleniyor..."
)

X = torch.stack([
    load_model_tensor(p)
    for p in test_df["file_path"]
]).to(DEVICE)

y_true = (
    test_df["true_num"]
    .to_numpy(dtype=int)
)

print(
    "Tensor:",
    tuple(X.shape),
    "|",
    X.device,
)

fold_rows = []

fold_probs_tta = []
fold_probs_no = []

# ---------------------------------------------------------
# 5 FOLD MODELİ TEK TEK TEST ET
# ---------------------------------------------------------

for fold, ckpt in checkpoints:
    print(
        f"\nFold {fold} modeli test ediliyor..."
    )

    model = V3Stronger().to(DEVICE)

    state = load_state(ckpt)

    model.load_state_dict(
        state
    )

    prob_no = predict_model(
        model,
        X,
        tta=False,
    )

    prob_tta = predict_model(
        model,
        X,
        tta=True,
    )

    fold_probs_no.append(
        prob_no
    )

    fold_probs_tta.append(
        prob_tta
    )

    m_no = metric_dict(
        y_true,
        prob_no,
    )

    m_tta = metric_dict(
        y_true,
        prob_tta,
    )

    temp = test_df.copy()

    temp[f"fold_{fold}_prob_no_tta"] = prob_no
    temp[f"fold_{fold}_prob_tta"] = prob_tta

    pair_no_df, pair_acc_no = pairwise_metric(
        temp,
        f"fold_{fold}_prob_no_tta",
    )

    pair_tta_df, pair_acc_tta = pairwise_metric(
        temp,
        f"fold_{fold}_prob_tta",
    )

    fold_rows.append({
        "fold": fold,
        "checkpoint": ckpt.name,

        "accuracy_no_tta": m_no["accuracy"],
        "real_correct_no_tta": m_no["real_correct"],
        "fake_correct_no_tta": m_no["fake_correct"],
        "macro_f1_no_tta": m_no["macro_f1"],
        "roc_auc_no_tta": m_no["roc_auc"],
        "pairwise_accuracy_no_tta": pair_acc_no,

        "accuracy_tta": m_tta["accuracy"],
        "real_correct_tta": m_tta["real_correct"],
        "fake_correct_tta": m_tta["fake_correct"],
        "macro_f1_tta": m_tta["macro_f1"],
        "roc_auc_tta": m_tta["roc_auc"],
        "pairwise_accuracy_tta": pair_acc_tta,
    })

    print(
        f"  NO-TTA: "
        f"{m_no['real_correct'] + m_no['fake_correct']}/24 "
        f"= %{100*m_no['accuracy']:.2f} | "
        f"REAL {m_no['real_correct']}/12 | "
        f"FAKE {m_no['fake_correct']}/12 | "
        f"PAIR {int(round(pair_acc_no*12))}/12"
    )

    print(
        f"  TTA   : "
        f"{m_tta['real_correct'] + m_tta['fake_correct']}/24 "
        f"= %{100*m_tta['accuracy']:.2f} | "
        f"REAL {m_tta['real_correct']}/12 | "
        f"FAKE {m_tta['fake_correct']}/12 | "
        f"PAIR {int(round(pair_acc_tta*12))}/12"
    )

    del model
    torch.cuda.empty_cache()

fold_summary = pd.DataFrame(
    fold_rows
)

fold_summary.to_csv(
    OUT_DIR
    / "01_fold_bazli_bagimsiz_test_sonuclari.csv",
    index=False,
    encoding="utf-8-sig",
)

# ---------------------------------------------------------
# 5-FOLD ENSEMBLE
# ---------------------------------------------------------

P_NO = np.stack(
    fold_probs_no,
    axis=0,
)

P_TTA = np.stack(
    fold_probs_tta,
    axis=0,
)

ensemble_no = P_NO.mean(
    axis=0
)

ensemble_tta = P_TTA.mean(
    axis=0
)

std_no = P_NO.std(
    axis=0
)

std_tta = P_TTA.std(
    axis=0
)

m_no = metric_dict(
    y_true,
    ensemble_no,
)

m_tta = metric_dict(
    y_true,
    ensemble_tta,
)

pred_no = m_no["pred"]
pred_tta = m_tta["pred"]

result = test_df.copy()

for i in range(5):
    result[
        f"fold_{i+1}_fake_prob_no_tta"
    ] = P_NO[i]

    result[
        f"fold_{i+1}_fake_prob_tta"
    ] = P_TTA[i]

result[
    "ensemble_fake_probability_no_tta"
] = ensemble_no

result[
    "ensemble_fake_probability_tta"
] = ensemble_tta

result[
    "fold_probability_std_no_tta"
] = std_no

result[
    "fold_probability_std_tta"
] = std_tta

result[
    "ensemble_prediction_no_tta"
] = [
    LABEL_NAME[x]
    for x in pred_no
]

result[
    "ensemble_prediction_tta"
] = [
    LABEL_NAME[x]
    for x in pred_tta
]

result[
    "correct_no_tta"
] = (
    result[
        "ensemble_prediction_no_tta"
    ]
    ==
    result["label"]
)

result[
    "correct_tta"
] = (
    result[
        "ensemble_prediction_tta"
    ]
    ==
    result["label"]
)

result[
    "true_class_confidence_tta"
] = np.where(
    result["label"] == "FAKE",
    ensemble_tta,
    1.0 - ensemble_tta,
)

result[
    "distance_to_threshold_tta"
] = np.abs(
    ensemble_tta - 0.5
)

result.to_csv(
    OUT_DIR
    / "02_24_gorsel_ensemble_detay.csv",
    index=False,
    encoding="utf-8-sig",
)

# Pairwise
pair_no_df, pair_acc_no = pairwise_metric(
    result,
    "ensemble_fake_probability_no_tta",
)

pair_tta_df, pair_acc_tta = pairwise_metric(
    result,
    "ensemble_fake_probability_tta",
)

pair_compare = pair_tta_df.rename(
    columns={
        "real_fake_probability":
        "real_fake_probability_tta",

        "fake_fake_probability":
        "fake_fake_probability_tta",

        "fake_minus_real_margin":
        "fake_minus_real_margin_tta",

        "pairwise_correct":
        "pairwise_correct_tta",
    }
)

pair_compare = pair_compare.merge(
    pair_no_df[
        [
            "pair_no",
            "real_fake_probability",
            "fake_fake_probability",
            "fake_minus_real_margin",
            "pairwise_correct",
        ]
    ].rename(
        columns={
            "real_fake_probability":
            "real_fake_probability_no_tta",

            "fake_fake_probability":
            "fake_fake_probability_no_tta",

            "fake_minus_real_margin":
            "fake_minus_real_margin_no_tta",

            "pairwise_correct":
            "pairwise_correct_no_tta",
        }
    ),
    on="pair_no",
    how="left",
)

pair_compare.to_csv(
    OUT_DIR
    / "03_12_cift_pairwise_detay.csv",
    index=False,
    encoding="utf-8-sig",
)

# ---------------------------------------------------------
# ALT GRUP ANALİZİ
# ---------------------------------------------------------

subgroup_rows = []

def add_subgroup(
    group_type,
    group_value,
    sub,
):
    if len(sub) == 0:
        return

    subgroup_rows.append({
        "group_type": group_type,
        "group_value": group_value,
        "n": len(sub),
        "accuracy_tta": float(
            sub["correct_tta"].mean()
        ),
        "accuracy_no_tta": float(
            sub["correct_no_tta"].mean()
        ),
        "mean_fake_probability_tta": float(
            sub[
                "ensemble_fake_probability_tta"
            ].mean()
        ),
        "mean_fold_std_tta": float(
            sub[
                "fold_probability_std_tta"
            ].mean()
        ),
    })

add_subgroup(
    "ALL",
    "ALL",
    result,
)

for value, sub in result.groupby(
    "label"
):
    add_subgroup(
        "label",
        value,
        sub,
    )

for value, sub in result.groupby(
    "format"
):
    add_subgroup(
        "format",
        value,
        sub,
    )

for value, sub in result[
    result["label"] == "FAKE"
].groupby("generator"):
    add_subgroup(
        "fake_generator",
        value,
        sub,
    )

subgroups = pd.DataFrame(
    subgroup_rows
)

subgroups.to_csv(
    OUT_DIR
    / "04_alt_grup_sonuclari.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# GRAFİK 1 — FOLD + ENSEMBLE
# =========================================================

xlabels = [
    f"Fold {i}"
    for i in range(1, 6)
] + ["5-Fold\nEnsemble"]

tta_vals = (
    100
    * fold_summary[
        "accuracy_tta"
    ].tolist()
    + [100 * m_tta["accuracy"]]
)

no_vals = (
    100
    * fold_summary[
        "accuracy_no_tta"
    ].tolist()
    + [100 * m_no["accuracy"]]
)

x = np.arange(
    len(xlabels)
)

w = 0.36

fig, ax = plt.subplots(
    figsize=(13, 7)
)

ax.bar(
    x - w/2,
    tta_vals,
    width=w,
    label="TTA",
)

ax.bar(
    x + w/2,
    no_vals,
    width=w,
    label="NO-TTA",
)

ax.set_xticks(
    x,
    xlabels,
)

ax.set_ylim(0, 105)

ax.set_ylabel(
    "Bağımsız test doğruluğu (%)"
)

ax.set_title(
    "24 Bağımsız Görüntü: 5 Fold Model ve Ensemble Karşılaştırması"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

label_bars(ax)

savefig(
    "05_foldlar_ve_ensemble_accuracy.png"
)

# =========================================================
# GRAFİK 2 — ENSEMBLE METRİKLER
# =========================================================

metric_names = [
    "Accuracy",
    "Precision",
    "FAKE Recall",
    "F1",
    "Macro-F1",
    "ROC-AUC",
]

tta_metric_vals = [
    100 * m_tta["accuracy"],
    100 * m_tta["precision"],
    100 * m_tta["recall_fake"],
    100 * m_tta["f1_fake"],
    100 * m_tta["macro_f1"],
    100 * m_tta["roc_auc"],
]

no_metric_vals = [
    100 * m_no["accuracy"],
    100 * m_no["precision"],
    100 * m_no["recall_fake"],
    100 * m_no["f1_fake"],
    100 * m_no["macro_f1"],
    100 * m_no["roc_auc"],
]

x = np.arange(
    len(metric_names)
)

w = 0.36

fig, ax = plt.subplots(
    figsize=(14, 7)
)

ax.bar(
    x - w/2,
    tta_metric_vals,
    width=w,
    label="TTA",
)

ax.bar(
    x + w/2,
    no_metric_vals,
    width=w,
    label="NO-TTA",
)

ax.set_xticks(
    x,
    metric_names,
)

ax.set_ylim(0, 105)

ax.set_ylabel("Skor (%)")

ax.set_title(
    "5-Fold Ensemble: Bağımsız Test Metrikleri"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

label_bars(ax)

savefig(
    "06_ensemble_metrikleri.png"
)

# =========================================================
# CONFUSION MATRICES
# =========================================================

plot_confusion(
    m_tta["cm"],
    (
        "5-Fold Ensemble — TTA\n"
        f"Accuracy %{100*m_tta['accuracy']:.1f}"
    ),
    "07_confusion_matrix_TTA.png",
)

plot_confusion(
    m_no["cm"],
    (
        "5-Fold Ensemble — NO-TTA\n"
        f"Accuracy %{100*m_no['accuracy']:.1f}"
    ),
    "08_confusion_matrix_NO_TTA.png",
)

# =========================================================
# GRAFİK 5 — REAL / FAKE BAŞARISI
# =========================================================

labels = [
    "REAL (12)",
    "FAKE (12)",
]

tta_class = [
    100 * m_tta["real_correct"] / 12,
    100 * m_tta["fake_correct"] / 12,
]

no_class = [
    100 * m_no["real_correct"] / 12,
    100 * m_no["fake_correct"] / 12,
]

x = np.arange(2)
w = 0.36

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.bar(
    x - w/2,
    tta_class,
    width=w,
    label="TTA",
)

ax.bar(
    x + w/2,
    no_class,
    width=w,
    label="NO-TTA",
)

ax.set_xticks(
    x,
    labels,
)

ax.set_ylim(0, 105)

ax.set_ylabel(
    "Doğru sınıflandırma (%)"
)

ax.set_title(
    "Bağımsız Test: REAL ve FAKE Sınıf Başarısı"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

label_bars(ax)

savefig(
    "09_real_fake_sinif_basarisi.png"
)

# =========================================================
# GRAFİK 6 — 24 GÖRÜNTÜ FAKE OLASILIĞI
# =========================================================

plot_df = result.copy()

plot_df["short_name"] = [
    f"P{int(p):02d}-{lab[0]}"
    for p, lab
    in zip(
        plot_df["pair_no"],
        plot_df["label"],
    )
]

x = np.arange(
    len(plot_df)
)

fig, ax = plt.subplots(
    figsize=(18, 8)
)

ax.bar(
    x,
    100
    * plot_df[
        "ensemble_fake_probability_tta"
    ],
)

ax.axhline(
    50,
    linestyle="--",
    linewidth=1.5,
    label="FAKE karar eşiği (%50)",
)

ax.set_xticks(
    x,
    plot_df["short_name"],
    rotation=45,
)

ax.set_ylim(0, 105)

ax.set_ylabel(
    "Ensemble FAKE olasılığı (%)"
)

ax.set_title(
    "24 Bağımsız Görüntü: Ensemble TTA Tahmin Olasılıkları\n"
    "Pxx-R = gerçek, Pxx-F = yapay"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.2,
)

savefig(
    "10_24_gorsel_fake_olasiliklari_TTA.png"
)

# =========================================================
# GRAFİK 7 — 12 ÇİFT PAIRWISE MARJ
# =========================================================

fig, ax = plt.subplots(
    figsize=(14, 7)
)

x = np.arange(
    1,
    13,
)

margins = (
    100
    * pair_tta_df[
        "fake_minus_real_margin"
    ].to_numpy()
)

ax.bar(
    x,
    margins,
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.5,
)

ax.set_xticks(
    x,
    [
        f"Çift {i}"
        for i in x
    ],
)

ax.set_ylabel(
    "FAKE olasılık farkı\n(Yapay - Gerçek, yüzde puan)"
)

ax.set_title(
    "12 Bağımsız Çift: Model Yapayı Gerçekten Daha Yapay Buldu mu? (TTA)"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

for i, value in enumerate(
    margins,
    start=1,
):
    ax.text(
        i,
        value + (
            1.2
            if value >= 0
            else -3.2
        ),
        "✓" if value > 0 else "✗",
        ha="center",
        fontsize=12,
    )

savefig(
    "11_12_cift_pairwise_marj_TTA.png"
)

# =========================================================
# GRAFİK 8 — PAIRWISE vs TEK TEK 24
# =========================================================

labels = [
    "24 görüntü\ntek tek",
    "12 çiftte\nhangisi yapay?",
]

tta_compare = [
    100 * m_tta["accuracy"],
    100 * pair_acc_tta,
]

no_compare = [
    100 * m_no["accuracy"],
    100 * pair_acc_no,
]

x = np.arange(2)
w = 0.36

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.bar(
    x - w/2,
    tta_compare,
    width=w,
    label="TTA",
)

ax.bar(
    x + w/2,
    no_compare,
    width=w,
    label="NO-TTA",
)

ax.set_xticks(
    x,
    labels,
)

ax.set_ylim(0, 105)

ax.set_ylabel(
    "Doğruluk (%)"
)

ax.set_title(
    "Bağımsız Test: Tekil Sınıflandırma ve Eşli Karşılaştırma"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25,
)

label_bars(ax)

savefig(
    "12_tekil24_vs_pairwise12.png"
)

# =========================================================
# GRAFİK 9 — FOLD BELİRSİZLİĞİ
# =========================================================

unc = result.sort_values(
    "fold_probability_std_tta",
    ascending=False,
)

x = np.arange(
    len(unc)
)

fig, ax = plt.subplots(
    figsize=(18, 8)
)

ax.bar(
    x,
    100
    * unc[
        "fold_probability_std_tta"
    ],
)

ax.set_xticks(
    x,
    [
        f"P{int(p):02d}-{l[0]}"
        for p, l
        in zip(
            unc["pair_no"],
            unc["label"],
        )
    ],
    rotation=45,
)

ax.set_ylabel(
    "5 fold FAKE olasılığı standart sapması (puan)"
)

ax.set_title(
    "Hangi Test Görüntülerinde 5 Fold Birbirinden Daha Fazla Ayrıştı?"
)

ax.grid(
    axis="y",
    alpha=0.25,
)

savefig(
    "13_fold_belirsizligi_gorsel_bazli.png"
)

# =========================================================
# FORMAT GRAFİĞİ
# =========================================================

format_acc = (
    result.groupby("format")[
        "correct_tta"
    ]
    .agg(["mean", "count"])
    .reset_index()
)

if len(format_acc) > 1:
    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.bar(
        format_acc["format"],
        100 * format_acc["mean"],
    )

    ax.set_ylim(0, 105)

    ax.set_ylabel(
        "TTA ensemble doğruluğu (%)"
    )

    ax.set_title(
        "Bağımsız Test: Format Bazlı Başarı"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    label_bars(ax)

    savefig(
        "14_format_bazli_basarim.png"
    )

# =========================================================
# GENERATOR GRAFİĞİ
# =========================================================

fake_only = result[
    result["label"] == "FAKE"
].copy()

gen_acc = (
    fake_only.groupby("generator")[
        "correct_tta"
    ]
    .agg(["mean", "count"])
    .reset_index()
)

if len(gen_acc) > 1:
    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.bar(
        gen_acc["generator"],
        100 * gen_acc["mean"],
    )

    ax.set_ylim(0, 105)

    ax.set_ylabel(
        "FAKE tespit doğruluğu (%)"
    )

    ax.set_title(
        "Bağımsız Test: Yapay Üretici Bazlı FAKE Tespiti (TTA)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    label_bars(ax)

    savefig(
        "15_generator_bazli_fake_tespiti.png"
    )

# =========================================================
# 408 OOF vs BAĞIMSIZ TEST
# =========================================================

OOF_TTA = MODEL_DIR / "OOF_ALL_408_TTA.csv"
OOF_NO = MODEL_DIR / "OOF_ALL_408_NO_TTA.csv"

oof_tta_acc = None
oof_no_acc = None

if OOF_TTA.exists():
    d = pd.read_csv(OOF_TTA)

    if "correct" in d.columns:
        c = d["correct"]

        if c.dtype != bool:
            c = (
                c.astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                })
            )

        oof_tta_acc = (
            100 * c.mean()
        )

if OOF_NO.exists():
    d = pd.read_csv(OOF_NO)

    if "correct" in d.columns:
        c = d["correct"]

        if c.dtype != bool:
            c = (
                c.astype(str)
                .str.lower()
                .map({
                    "true": True,
                    "false": False,
                })
            )

        oof_no_acc = (
            100 * c.mean()
        )

if (
    oof_tta_acc is not None
    and
    oof_no_acc is not None
):
    labels = [
        "408 OOF\nTTA",
        "408 OOF\nNO-TTA",
        "Bağımsız 24\nEnsemble TTA",
        "Bağımsız 24\nEnsemble NO-TTA",
    ]

    values = [
        oof_tta_acc,
        oof_no_acc,
        100 * m_tta["accuracy"],
        100 * m_no["accuracy"],
    ]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.bar(
        labels,
        values,
    )

    ax.set_ylim(0, 105)

    ax.set_ylabel(
        "Doğruluk (%)"
    )

    ax.set_title(
        "408 Group-Aware OOF ve 24 Bağımsız Test Karşılaştırması"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    label_bars(ax)

    savefig(
        "16_408_oof_vs_bagimsiz24.png"
    )

# =========================================================
# HATA MONTAJLARI
# =========================================================

make_error_montage(
    result,
    "ensemble_fake_probability_tta",
    "ensemble_prediction_tta",
    "17_hatalar_montaj_TTA.png",
    "5-Fold Ensemble TTA — Yanlış Sınıflandırılan Bağımsız Test Görselleri",
)

make_error_montage(
    result,
    "ensemble_fake_probability_no_tta",
    "ensemble_prediction_no_tta",
    "18_hatalar_montaj_NO_TTA.png",
    "5-Fold Ensemble NO-TTA — Yanlış Sınıflandırılan Bağımsız Test Görselleri",
)

# =========================================================
# EN ZOR / EN KOLAY GÖRSELLER CSV
# =========================================================

difficulty = result[
    [
        "test_id",
        "pair_no",
        "label",
        "format",
        "generator",
        "file_name",
        "ensemble_fake_probability_tta",
        "ensemble_prediction_tta",
        "correct_tta",
        "true_class_confidence_tta",
        "distance_to_threshold_tta",
        "fold_probability_std_tta",
    ]
].copy()

difficulty = difficulty.sort_values(
    [
        "correct_tta",
        "true_class_confidence_tta",
    ],
    ascending=[
        True,
        True,
    ],
)

difficulty.to_csv(
    OUT_DIR
    / "19_gorsel_zorluk_siralamasi_TTA.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# ÖZET METRİK CSV
# =========================================================

summary = pd.DataFrame([
    {
        "mode": "5-fold ensemble NO-TTA",
        "correct_24": (
            m_no["real_correct"]
            + m_no["fake_correct"]
        ),
        "total_24": 24,
        "accuracy": m_no["accuracy"],
        "real_correct_12": m_no["real_correct"],
        "fake_correct_12": m_no["fake_correct"],
        "precision_fake": m_no["precision"],
        "recall_fake": m_no["recall_fake"],
        "f1_fake": m_no["f1_fake"],
        "macro_f1": m_no["macro_f1"],
        "roc_auc": m_no["roc_auc"],
        "pairwise_correct_12": int(
            pair_no_df[
                "pairwise_correct"
            ].sum()
        ),
        "pairwise_accuracy": pair_acc_no,
    },
    {
        "mode": "5-fold ensemble TTA",
        "correct_24": (
            m_tta["real_correct"]
            + m_tta["fake_correct"]
        ),
        "total_24": 24,
        "accuracy": m_tta["accuracy"],
        "real_correct_12": m_tta["real_correct"],
        "fake_correct_12": m_tta["fake_correct"],
        "precision_fake": m_tta["precision"],
        "recall_fake": m_tta["recall_fake"],
        "f1_fake": m_tta["f1_fake"],
        "macro_f1": m_tta["macro_f1"],
        "roc_auc": m_tta["roc_auc"],
        "pairwise_correct_12": int(
            pair_tta_df[
                "pairwise_correct"
            ].sum()
        ),
        "pairwise_accuracy": pair_acc_tta,
    },
])

summary.to_csv(
    OUT_DIR
    / "20_GENEL_METRIKLER.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# OTOMATİK YORUM RAPORU
# =========================================================

lines = []

lines.append(
    "CNN-v3 STRONGER — 24 BAĞIMSIZ TEST / 5-FOLD ENSEMBLE"
)

lines.append("=" * 72)
lines.append("")

lines.append(
    "NO-TTA ENSEMBLE"
)

lines.append(
    f"Doğru: "
    f"{m_no['real_correct'] + m_no['fake_correct']}/24 "
    f"(%{100*m_no['accuracy']:.2f})"
)

lines.append(
    f"REAL: {m_no['real_correct']}/12"
)

lines.append(
    f"FAKE: {m_no['fake_correct']}/12"
)

lines.append(
    f"Macro-F1: {m_no['macro_f1']:.4f}"
)

lines.append(
    f"ROC-AUC: {m_no['roc_auc']:.4f}"
)

lines.append(
    f"Pairwise: "
    f"{int(pair_no_df['pairwise_correct'].sum())}/12 "
    f"(%{100*pair_acc_no:.2f})"
)

lines.append("")

lines.append(
    "TTA ENSEMBLE"
)

lines.append(
    f"Doğru: "
    f"{m_tta['real_correct'] + m_tta['fake_correct']}/24 "
    f"(%{100*m_tta['accuracy']:.2f})"
)

lines.append(
    f"REAL: {m_tta['real_correct']}/12"
)

lines.append(
    f"FAKE: {m_tta['fake_correct']}/12"
)

lines.append(
    f"Macro-F1: {m_tta['macro_f1']:.4f}"
)

lines.append(
    f"ROC-AUC: {m_tta['roc_auc']:.4f}"
)

lines.append(
    f"Pairwise: "
    f"{int(pair_tta_df['pairwise_correct'].sum())}/12 "
    f"(%{100*pair_acc_tta:.2f})"
)

lines.append("")

# TTA gain
tta_gain = (
    100
    * (
        m_tta["accuracy"]
        - m_no["accuracy"]
    )
)

if tta_gain > 0:
    lines.append(
        f"TTA bağımsız testte doğruluğu "
        f"{tta_gain:.2f} yüzde puan artırdı."
    )

elif tta_gain < 0:
    lines.append(
        f"TTA bağımsız testte doğruluğu "
        f"{abs(tta_gain):.2f} yüzde puan düşürdü."
    )

else:
    lines.append(
        "TTA bağımsız test doğruluğunu değiştirmedi."
    )

# Class bias
real_acc = (
    m_tta["real_correct"] / 12
)

fake_acc = (
    m_tta["fake_correct"] / 12
)

if abs(real_acc - fake_acc) < 0.10:
    lines.append(
        "TTA ensemble REAL ve FAKE sınıflarında görece dengeli."
    )

elif real_acc > fake_acc:
    lines.append(
        "TTA ensemble REAL sınıfında FAKE sınıfına göre daha başarılı."
    )

else:
    lines.append(
        "TTA ensemble FAKE sınıfında REAL sınıfına göre daha başarılı."
    )

# Pair vs individual
if pair_acc_tta > m_tta["accuracy"]:
    lines.append(
        "Eşli 'hangisi daha yapay?' görevi, "
        "24 görüntüyü tek tek sınıflandırmaktan daha kolay göründü."
    )

elif pair_acc_tta < m_tta["accuracy"]:
    lines.append(
        "Tek tek sınıflandırma, eşli karşılaştırmadan daha başarılı çıktı."
    )

else:
    lines.append(
        "Tekil ve pairwise başarı aynı seviyede çıktı."
    )

# Fold variability
fold_acc_tta = (
    100
    * fold_summary[
        "accuracy_tta"
    ].to_numpy()
)

lines.append(
    f"Tek fold modellerinin TTA doğruluk ortalaması: "
    f"%{fold_acc_tta.mean():.2f} "
    f"(std={fold_acc_tta.std(ddof=0):.2f} puan)."
)

lines.append(
    f"5-fold ensemble TTA doğruluğu: "
    f"%{100*m_tta['accuracy']:.2f}."
)

# 408 comparison
if oof_tta_acc is not None:
    gap = (
        100 * m_tta["accuracy"]
        - oof_tta_acc
    )

    lines.append(
        f"408 görüntü OOF TTA: %{oof_tta_acc:.2f}; "
        f"24 bağımsız test ensemble TTA: %{100*m_tta['accuracy']:.2f}; "
        f"fark: {gap:+.2f} yüzde puan."
    )

# Hard examples
lines.append("")
lines.append(
    "TTA ENSEMBLE — EN ZOR / BELİRSİZ GÖRÜNTÜLER"
)

hard = result.sort_values(
    [
        "correct_tta",
        "true_class_confidence_tta",
    ],
    ascending=[
        True,
        True,
    ],
).head(5)

for _, r in hard.iterrows():
    lines.append(
        f"Pair {int(r['pair_no']):02d} {r['label']} | "
        f"tahmin={r['ensemble_prediction_tta']} | "
        f"FAKE olasılığı=%{100*r['ensemble_fake_probability_tta']:.1f} | "
        f"fold std={100*r['fold_probability_std_tta']:.1f} puan | "
        f"{'DOĞRU' if r['correct_tta'] else 'YANLIŞ'}"
    )

# Format
lines.append("")
lines.append(
    "FORMAT BAZLI TTA"
)

for fmt, g in result.groupby("format"):
    lines.append(
        f"{fmt}: "
        f"{int(g['correct_tta'].sum())}/{len(g)} "
        f"(%{100*g['correct_tta'].mean():.1f})"
    )

# Generator
lines.append("")
lines.append(
    "FAKE GENERATOR BAZLI TTA"
)

for gen, g in fake_only.groupby("generator"):
    lines.append(
        f"{gen}: "
        f"{int(g['correct_tta'].sum())}/{len(g)} "
        f"(%{100*g['correct_tta'].mean():.1f})"
    )

lines.append("")
lines.append(
    "NOT: Bu test seti model geliştirme/hiperparametre seçimi için "
    "kullanılmamalı; mevcut modelin bağımsız dış test sonucu olarak saklanmalıdır."
)

(OUT_DIR / "21_BAGIMSIZ_TEST_RAPORU.txt").write_text(
    "\n".join(lines),
    encoding="utf-8",
)

# =========================================================
# TERMİNAL ÖZET
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "24 BAĞIMSIZ TEST — 5-FOLD ENSEMBLE TAMAMLANDI"
)

print(
    "=" * 78
)

print(
    "\nNO-TTA:"
)

print(
    f"  Doğru : "
    f"{m_no['real_correct'] + m_no['fake_correct']}/24 "
    f"(%{100*m_no['accuracy']:.2f})"
)

print(
    f"  REAL  : {m_no['real_correct']}/12"
)

print(
    f"  FAKE  : {m_no['fake_correct']}/12"
)

print(
    f"  Pair  : "
    f"{int(pair_no_df['pairwise_correct'].sum())}/12 "
    f"(%{100*pair_acc_no:.2f})"
)

print(
    "\nTTA:"
)

print(
    f"  Doğru : "
    f"{m_tta['real_correct'] + m_tta['fake_correct']}/24 "
    f"(%{100*m_tta['accuracy']:.2f})"
)

print(
    f"  REAL  : {m_tta['real_correct']}/12"
)

print(
    f"  FAKE  : {m_tta['fake_correct']}/12"
)

print(
    f"  Pair  : "
    f"{int(pair_tta_df['pairwise_correct'].sum())}/12 "
    f"(%{100*pair_acc_tta:.2f})"
)

print(
    "\nÇıktı klasörü:"
)

print(
    OUT_DIR
)
