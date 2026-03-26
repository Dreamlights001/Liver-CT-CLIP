"""
CT-CLIP Zero-Shot Module

This module provides:
1. CTClipInference - Zero-shot classification with CT-CLIP
2. Regression utilities - HCC necrosis rate prediction

"""

import hashlib
import json
import math
import os
import random
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from ct_clip import CTCLIP
from data_inference_nii import CTReportDatasetinfer
from eval import evaluate_internal
from src.models.utils import cosine_lr
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, BertModel, BertTokenizer


# =============================================================================
# Helper Functions (shared)
# =============================================================================

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


# =============================================================================
# Regression Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_proxy_env():
    """Disable socks proxy to avoid HuggingFace errors."""
    for key in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(key, "").startswith("socks://"):
            os.environ.pop(key, None)


def safe_float(x, default=None):
    """Safely convert to float."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def parse_folder_name(name: str):
    """Parse folder name like ID01.02555191 -> (1, 2555191)"""
    if not name.startswith("ID") or "." not in name:
        return None, None
    id_part, name_part = name[2:].split(".", 1)
    try:
        id_int = int(id_part)
    except Exception:
        id_int = None
    try:
        name_int = int(name_part)
    except Exception:
        name_int = None
    return id_int, name_int


TARGET_LEAKAGE_COLS = {"坏死比例分组", "坏死比例"}
TUMOR_MARKER_PRIMARY_COLS = (
    "手术切除前AFP",
    "手术切除前 PIVKA",
    "诊断时AFP",
    "诊断时PIVKA-II",
)
TUMOR_MARKER_KEYWORDS = ("afp", "pivka", "甲胎蛋白")
AGE_COL_KEYWORDS = ("年龄", "age")
SEX_COL_KEYWORDS = ("性别", "gender", "sex")

# Prompt templates for ablation experiments.
# Template-1: arterial_only
# Template-2: arterial_portal
# Template-3: all_features
# Template-4: tumor_markers_text_only
PROMPT_TEMPLATE_DESCRIPTIONS = {
    "arterial_only": "Template-1: arterial-only CT with professional imaging prompt + minimal demographics (age/sex).",
    "arterial_portal": "Template-2: arterial+portal CT with professional dual-phase prompt + minimal demographics (age/sex).",
    "all_features": "Template-3: arterial+portal CT + all non-target clinical fields with professional multimodal prompt.",
    "tumor_markers_text_only": "Template-4: tumor-marker text only (no CT sequence) with professional lab-focused prompt.",
}


def normalize_for_text(value):
    """Normalize table value for prompt text."""
    if pd.isna(value):
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return f"{value:.6g}"
    return str(value).strip()


def normalize_col_name_for_match(col_name):
    """Normalize column name for robust matching."""
    text = str(col_name).strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def is_tumor_marker_column(col_name):
    """Whether a column should be treated as tumor-marker related."""
    raw = str(col_name).strip()
    norm = normalize_col_name_for_match(raw)
    primary_norm = {normalize_col_name_for_match(c) for c in TUMOR_MARKER_PRIMARY_COLS}
    if norm in primary_norm:
        return True
    return any(keyword in norm for keyword in TUMOR_MARKER_KEYWORDS)


def select_tumor_marker_columns(row_dict, target_col):
    """Select tumor marker columns in stable order."""
    excluded = set(TARGET_LEAKAGE_COLS)
    excluded.add(target_col)

    all_cols = [col for col in row_dict.keys() if col not in excluded]
    marker_cols = [col for col in all_cols if is_tumor_marker_column(col)]

    if not marker_cols:
        return []

    # Keep a deterministic order: primary columns first, then any extra marker columns.
    primary_order = {normalize_col_name_for_match(c): i for i, c in enumerate(TUMOR_MARKER_PRIMARY_COLS)}
    marker_cols_sorted = sorted(
        marker_cols,
        key=lambda c: (primary_order.get(normalize_col_name_for_match(c), 999), all_cols.index(c)),
    )
    return marker_cols_sorted


def select_minimal_background_columns(row_dict, target_col):
    """Select only minimal non-leakage background fields for template-1/2 (age/sex)."""
    excluded = set(TARGET_LEAKAGE_COLS)
    excluded.add(target_col)

    def find_first_col(keywords):
        for col, val in row_dict.items():
            if col in excluded:
                continue
            if normalize_for_text(val) is None:
                continue
            norm = normalize_col_name_for_match(col)
            if any(keyword in norm for keyword in keywords):
                return col
        return None

    cols = [find_first_col(AGE_COL_KEYWORDS), find_first_col(SEX_COL_KEYWORDS)]
    # keep order and uniqueness
    selected = []
    for col in cols:
        if col is not None and col not in selected:
            selected.append(col)
    return selected


def row_to_text(row_dict, include_cols, include_missing=False, missing_token="缺失"):
    """Convert selected columns to text format."""
    parts = []
    for col in include_cols:
        val = normalize_for_text(row_dict.get(col))
        if val is None or val == "":
            if include_missing:
                parts.append(f"{col}:{missing_token}")
            continue
        parts.append(f"{col}:{val}")
    return "; ".join(parts)


def build_field_sentence(col_name, value):
    """Render one field into a professional sentence."""
    norm = normalize_col_name_for_match(col_name)
    if any(keyword in norm for keyword in AGE_COL_KEYWORDS):
        return f"患者年龄为{value}岁。"
    if any(keyword in norm for keyword in SEX_COL_KEYWORDS):
        return f"患者性别编码记录为{value}。"
    if is_tumor_marker_column(col_name):
        return f"肿瘤标志物{col_name}为{value}。"
    return f"临床记录显示{col_name}为{value}。"


def render_field_narrative(row_dict, columns):
    """Render selected columns into full-sentence narrative text."""
    sentences = []
    for col in columns:
        val = normalize_for_text(row_dict.get(col))
        if val is None or val == "":
            continue
        sentences.append(build_field_sentence(col, val))
    return " ".join(sentences)


def render_minimal_background_narrative(row_dict, target_col):
    """Render minimal background (age/sex) as narrative sentences."""
    cols = select_minimal_background_columns(row_dict, target_col)
    return render_field_narrative(row_dict, cols)


def render_all_features_narrative(row_dict, target_col):
    """Render all non-target features as narrative sentences."""
    cols = select_feature_columns(row_dict, "all_features", target_col)
    return render_field_narrative(row_dict, cols)


def render_tumor_markers_narrative(row_dict, target_col):
    """Render tumor-marker-only fields as narrative sentences."""
    cols = select_tumor_marker_columns(row_dict, target_col)
    return render_field_narrative(row_dict, cols)


def select_feature_columns(row_dict, prompt_template, target_col):
    """Select feature columns according to prompt template."""
    if prompt_template == "tumor_markers_text_only":
        return select_tumor_marker_columns(row_dict, target_col)
    if prompt_template in ("arterial_only", "arterial_portal"):
        return select_minimal_background_columns(row_dict, target_col)

    excluded = set(TARGET_LEAKAGE_COLS)
    excluded.add(target_col)
    valid_cols = []
    for col, val in row_dict.items():
        if col in excluded:
            continue
        if normalize_for_text(val) is None:
            continue
        valid_cols.append(col)

    if prompt_template == "all_features":
        selected = valid_cols
    else:
        raise ValueError(f"Unknown prompt template: {prompt_template}")
    return selected


def scan_phase(scan_path):
    """Infer scan phase from file name.

    HCC convention in this project:
    - 1.nii.gz: arterial phase
    - 2.nii.gz: portal venous phase
    """
    name = Path(scan_path).name.lower()
    if "动脉" in name or "arterial" in name:
        return "arterial"
    if "门脉" in name or "静脉" in name or "portal" in name or "venous" in name:
        return "portal"
    if name == "1.nii.gz" or re.search(r"(^|[_\-])1\.nii\.gz$", name):
        return "arterial"
    if name == "2.nii.gz" or re.search(r"(^|[_\-])2\.nii\.gz$", name):
        return "portal"
    return "unknown"


def select_scans_for_patient(scans, prompt_template):
    """Select scans to use for this patient according to template."""
    if prompt_template == "tumor_markers_text_only":
        return []
    if prompt_template != "arterial_only":
        return scans
    arterial_scans = [s for s in scans if scan_phase(s) == "arterial"]
    if arterial_scans:
        return arterial_scans
    return scans[:1]


def build_prompt_text(row_dict, prompt_template, target_col, phase=None):
    """Build one prompt text with a configurable template."""
    phase_map = {
        "arterial": "动脉期",
        "portal": "门静脉期",
        "unknown": "未知期",
        None: "未指定期别",
    }
    phase_text = phase_map.get(phase, "未指定期别")

    if prompt_template == "arterial_only":
        background_text = render_minimal_background_narrative(row_dict, target_col)
        background_clause = f"患者背景方面，{background_text}" if background_text else ""
        return (
            "任务: 基于术前肝脏增强CT评估肿瘤坏死情况，输出坏死比例分组与坏死比例。"
            f"输入设置: 当前输入为{phase_text}CT，模板仅使用动脉期序列。"
            "器官定位前提: 先确认影像已覆盖肝脏解剖区域，并在肝实质内完成病灶评估，避免被非肝脏结构干扰。"
            "影像判读时重点评估病灶动脉期强化强度与均匀性、强化缺损区范围、病灶边界及内部无强化区比例，"
            "据此判断是否接近完全坏死。"
            f"{background_clause}"
            "输出约束: 优先判断完全坏死分组(0/1)，再给出0-1坏死比例。"
        )
    if prompt_template == "arterial_portal":
        background_text = render_minimal_background_narrative(row_dict, target_col)
        background_clause = f"患者背景方面，{background_text}" if background_text else ""
        return (
            "任务: 基于术前双期增强CT(动脉期+门静脉期)联合评估肿瘤坏死情况，输出坏死比例分组与坏死比例。"
            f"输入设置: 当前输入为{phase_text}CT，模型将联合同一患者另一期序列进行综合判断。"
            "器官定位前提: 先确认病灶位于肝脏实质，再比较双期强化差异，避免将邻近脏器强化误判为肿瘤活性。"
            "影像判读需对照动脉期与门静脉期的强化持续/洗脱模式、非强化坏死区范围、病灶异质性及潜在活性残留。"
            f"{background_clause}"
            "输出约束: 优先判断完全坏死分组(0/1)，再给出0-1坏死比例。"
        )
    if prompt_template == "all_features":
        clinical_text = render_all_features_narrative(row_dict, target_col)
        clinical_clause = f"结构化临床信息方面，{clinical_text}" if clinical_text else ""
        return (
            "任务: 基于术前增强CT与结构化临床信息进行多模态评估，输出坏死比例分组与坏死比例。"
            f"输入设置: 当前输入为{phase_text}CT。"
            "器官定位前提: 先锁定肝脏解剖区域并识别肝内病灶，再结合临床信息推断坏死状态。"
            "评估时应综合影像强化模式、坏死区比例、病灶异质性以及临床背景(含肿瘤标志物、肝功能与治疗相关信息)进行联合判断。"
            "特征约束: 仅使用非目标字段(已排除坏死比例分组和坏死比例本身)。"
            f"{clinical_clause}"
            "输出约束: 优先判断完全坏死分组(0/1)，再给出0-1坏死比例。"
        )
    if prompt_template == "tumor_markers_text_only":
        marker_text = render_tumor_markers_narrative(row_dict, target_col)
        marker_clause = f"可用肿瘤标志物信息如下: {marker_text}" if marker_text else "当前无可用肿瘤标志物字段。"
        return (
            "任务: 在不使用CT影像的前提下，仅基于肿瘤标志物文本信息评估肿瘤坏死情况，输出坏死比例分组与坏死比例。"
            "判读要点: 重点结合AFP/PIVKA相关指标的绝对水平与相对差异，评估肿瘤活性残留风险。"
            f"{marker_clause}"
            "输出约束: 优先判断完全坏死分组(0/1)，再给出0-1坏死比例。"
        )
    raise ValueError(f"Unknown prompt template: {prompt_template}")


def canonical_scan_handling(scan_handling):
    """Map legacy scan handling args to current patient-level modes."""
    if scan_handling == "separate":
        return "distinguish"
    if scan_handling == "average":
        return "ignore"
    return scan_handling


def tokenize_text(tokenizer, text, device, batch_size=1):
    """Tokenize prompt text. Supports one string or list of strings."""
    if isinstance(text, str):
        texts = [text] * max(1, int(batch_size))
    else:
        texts = list(text)
    tokens_dict = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True, max_length=512
    )
    tokens_dict = {k: v.to(device) for k, v in tokens_dict.items()}
    return BatchEncoding(tokens_dict)


def slugify(value):
    """Build filesystem-safe slug."""
    raw_text = str(value).strip().lower()
    text = raw_text
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    if text:
        return text
    return f"u{hashlib.md5(raw_text.encode('utf-8')).hexdigest()[:8]}"


def train_split_tag(train_n, train_ratio):
    """Build short split tag for output directory naming."""
    if train_n is not None:
        return f"n{int(train_n)}"
    if train_ratio is not None:
        return f"r{str(train_ratio).replace('.', 'p')}"
    return "default"


def resolve_regression_out_dir(args):
    """Resolve regression output directory from config."""
    base_dir = Path(args.out_dir)
    if not args.auto_out_subdir:
        return base_dir, None

    if args.run_name:
        subdir = slugify(args.run_name)
    else:
        subdir = "__".join(
            [
                f"target-{slugify(args.target_col)}",
                f"tmpl-{slugify(args.prompt_template)}",
                f"scan-{slugify(args.scan_handling)}",
                f"split-{slugify(train_split_tag(args.train_n, args.train_ratio))}",
                f"text-{int(bool(args.use_text))}",
                f"seed-{int(args.seed)}",
            ]
        )
    return base_dir / subdir, subdir


def normalize_target(values):
    """Normalize target values to 0-1 range."""
    vals = [v for v in values if v is not None]
    if not vals:
        return values, 1.0
    vmax = max(vals)
    scale = 1.0
    # Heuristic: if values look like 0-100, scale to 0-1
    if vmax > 1.5 and vmax <= 100.0:
        scale = 100.0
    return [v / scale if v is not None else None for v in values], scale


def resize_array(array, current_spacing, target_spacing):
    """Resize array to match target spacing."""
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    return resized_array


def apply_liver_prior_crop(volume_hwd, crop_mode):
    """Apply non-segmentation organ-prior crop on HWD volume."""
    if crop_mode in (None, "none"):
        return volume_hwd

    h, w, d = volume_hwd.shape
    if crop_mode == "right_upper_abdomen":
        # Heuristic liver prior region: upper half + right-biased abdomen.
        h0, h1 = int(0.05 * h), int(0.72 * h)
        w0, w1 = int(0.28 * w), int(0.98 * w)
        d0, d1 = int(0.12 * d), int(0.96 * d)
        return volume_hwd[h0:h1, w0:w1, d0:d1]
    if crop_mode == "non_liver_background":
        # Pseudo-negative region for Stage-0: left-lower abdomen.
        h0, h1 = int(0.30 * h), int(0.98 * h)
        w0, w1 = int(0.00 * w), int(0.62 * w)
        d0, d1 = int(0.05 * d), int(0.88 * d)
        return volume_hwd[h0:h1, w0:w1, d0:d1]
    if crop_mode == "global_center":
        h0, h1 = int(0.05 * h), int(0.95 * h)
        w0, w1 = int(0.05 * w), int(0.95 * w)
        d0, d1 = int(0.05 * d), int(0.95 * d)
        return volume_hwd[h0:h1, w0:w1, d0:d1]
    raise ValueError(f"Unknown crop mode: {crop_mode}")


def apply_liver_window_fusion(hu_volume, enabled=True):
    """Fuse global CT intensity with liver window to emphasize liver parenchyma."""
    global_norm = np.clip(hu_volume, -1000.0, 1000.0) / 1000.0
    if not enabled:
        return global_norm.astype(np.float32)
    wl, ww = 60.0, 150.0
    low, high = wl - ww / 2.0, wl + ww / 2.0
    liver_norm = np.clip((hu_volume - low) / max(1e-6, high - low), 0.0, 1.0)
    liver_norm = (liver_norm * 2.0) - 1.0
    fused = 0.45 * global_norm + 0.55 * liver_norm
    return np.clip(fused, -1.0, 1.0).astype(np.float32)


def robust_phase_normalize(volume_hwd, phase, enabled=True):
    """Phase-aware robust normalization to reduce arterial/portal shift."""
    if not enabled:
        return volume_hwd.astype(np.float32)
    flat = volume_hwd.reshape(-1)
    if flat.size == 0:
        return volume_hwd.astype(np.float32)
    lo, hi = np.percentile(flat, [1.0, 99.0])
    scale = max(1e-4, hi - lo)
    out = (volume_hwd - lo) / scale
    out = (out * 2.0) - 1.0
    clip_val = 2.8
    if phase == "arterial":
        clip_val = 2.6
    elif phase == "portal":
        clip_val = 3.0
    out = np.clip(out, -clip_val, clip_val)
    out = out / clip_val
    return out.astype(np.float32)


def nii_img_to_tensor_regression(path, target_depth=240, preprocess_cfg=None, phase=None, crop_mode_override=None):
    """Convert NIfTI file to tensor for regression with non-segmentation liver priors."""
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    zooms = nii_img.header.get_zooms()
    xy_spacing = safe_float(zooms[0], 1.0)
    z_spacing = safe_float(zooms[2], 1.0) if len(zooms) > 2 else 1.0

    slope = safe_float(nii_img.header.get("scl_slope"), 1.0)
    intercept = safe_float(nii_img.header.get("scl_inter"), 0.0)
    if slope is None:
        slope = 1.0
    if intercept is None:
        intercept = 0.0

    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1200.0, 1200.0
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    crop_mode = crop_mode_override
    if crop_mode is None and preprocess_cfg is not None:
        crop_mode = getattr(preprocess_cfg, "liver_prior_crop", "right_upper_abdomen")
    if crop_mode is None:
        crop_mode = "none"
    img_data = apply_liver_prior_crop(img_data, crop_mode=crop_mode)

    liver_window_enabled = True
    phase_norm_enabled = True
    if preprocess_cfg is not None:
        liver_window_enabled = bool(getattr(preprocess_cfg, "liver_window", True))
        phase_norm_enabled = bool(getattr(preprocess_cfg, "phase_norm", True))

    img_data = apply_liver_window_fusion(img_data, enabled=liver_window_enabled)
    img_data = robust_phase_normalize(img_data, phase=phase, enabled=phase_norm_enabled)

    tensor = torch.tensor(img_data)

    target_shape = (480, 480, target_depth)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = F.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    return tensor


@dataclass
class Sample:
    """Sample for regression dataset."""
    patient_key: str
    features: dict
    ratio_target: float
    group_target: float
    scans: list


class HCCDataset(Dataset):
    """Patient-level dataset for HCC regression."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.patient_key, s.features, s.ratio_target, s.group_target, s.scans


@dataclass
class LiverAdaptSample:
    """One Stage-0 liver-awareness sample."""
    scan_path: str
    label: float
    crop_mode: str
    phase: Optional[str]


class LiverAdaptDataset(Dataset):
    """Dataset for Stage-0 liver-awareness warm-up."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s.scan_path, s.label, s.crop_mode, s.phase


class LiverAwareAdapter(nn.Module):
    """Image-latent adapter to improve organ-awareness before downstream multitask."""
    def __init__(self, clip, hidden=256, dropout=0.2):
        super().__init__()
        self.clip = clip
        self.adapter = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, text_tokens, images, device):
        _, image_latents, _ = self.clip(text_tokens, images, device=device, return_latents=True)
        adapted = image_latents + 0.2 * self.adapter(image_latents)
        return self.classifier(adapted).squeeze(1)


def collect_nii_paths(root):
    """Collect NIfTI paths recursively under root."""
    if root is None:
        return []
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(str(p) for p in root_path.rglob("*.nii.gz"))


def build_liver_adapt_samples(train_samples, args):
    """Build Stage-0 samples with positives from liver CT and negatives from external or pseudo non-liver crops."""
    positives = []
    for s in train_samples:
        selected = select_scans_for_patient(s.scans, args.prompt_template)
        if not selected:
            selected = s.scans
        for scan in selected:
            positives.append(
                LiverAdaptSample(
                    scan_path=scan,
                    label=1.0,
                    crop_mode=args.liver_prior_crop,
                    phase=scan_phase(scan),
                )
            )

    external_neg_paths = collect_nii_paths(args.stage0_negative_root)
    if external_neg_paths:
        random.Random(args.seed).shuffle(external_neg_paths)
        cap = min(len(external_neg_paths), max(1, args.stage0_max_negatives), max(1, len(positives)))
        external_neg_paths = external_neg_paths[:cap]

    negatives = []
    for scan in external_neg_paths:
        negatives.append(
            LiverAdaptSample(
                scan_path=scan,
                label=0.0,
                crop_mode="global_center",
                phase=scan_phase(scan),
            )
        )

    if not negatives and args.stage0_use_pseudo_negatives:
        for p in positives:
            negatives.append(
                LiverAdaptSample(
                    scan_path=p.scan_path,
                    label=0.0,
                    crop_mode="non_liver_background",
                    phase=p.phase,
                )
            )

    samples = positives + negatives
    random.Random(args.seed).shuffle(samples)
    return samples, len(positives), len(external_neg_paths), len(negatives)


def stage0_collate_fn(batch):
    """Collate Stage-0 batches."""
    scan_paths, labels, crop_modes, phases = zip(*batch)
    return list(scan_paths), list(labels), list(crop_modes), list(phases)


def _unfreeze_last_n_blocks(transformer_module, last_n):
    """Unfreeze last-n blocks in a transformer-like module with .layers."""
    if not hasattr(transformer_module, "layers"):
        return 0, 0
    layers = transformer_module.layers
    total = len(layers)
    if total == 0:
        return 0, 0
    if last_n <= 0:
        return 0, total
    n = min(last_n, total)
    for block in layers[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    return n, total


def configure_stage0_trainable(clip, unfreeze_last_n):
    """Configure trainable parameters for Stage-0 warm-up using a single ablation knob."""
    for p in clip.parameters():
        p.requires_grad = False

    # Always tune visual projection for domain adaptation.
    projection_params = 0
    for p in clip.to_visual_latent.parameters():
        p.requires_grad = True
        projection_params += p.numel()

    vt = clip.visual_transformer
    if unfreeze_last_n is None:
        unfreeze_last_n = 1
    if unfreeze_last_n < 0:
        for p in clip.visual_transformer.parameters():
            p.requires_grad = True
        spatial_total = len(getattr(getattr(vt, "enc_spatial_transformer", None), "layers", []))
        temporal_total = len(getattr(getattr(vt, "enc_temporal_transformer", None), "layers", []))
        spatial_unfrozen = spatial_total
        temporal_unfrozen = temporal_total
    else:
        spatial_unfrozen, spatial_total = _unfreeze_last_n_blocks(
            getattr(vt, "enc_spatial_transformer", None), unfreeze_last_n
        )
        temporal_unfrozen, temporal_total = _unfreeze_last_n_blocks(
            getattr(vt, "enc_temporal_transformer", None), unfreeze_last_n
        )

    return {
        "unfreeze_last_n": int(unfreeze_last_n),
        "projection_params": int(projection_params),
        "spatial_unfrozen_blocks": int(spatial_unfrozen),
        "spatial_total_blocks": int(spatial_total),
        "temporal_unfrozen_blocks": int(temporal_unfrozen),
        "temporal_total_blocks": int(temporal_total),
    }


def run_stage0_liver_adapt(clip, tokenizer, train_samples, args, device, out_dir):
    """Stage-0 warm-up to make visual branch liver-aware without segmentation."""
    samples, num_pos, num_external_negs, num_total_negs = build_liver_adapt_samples(train_samples, args)
    if num_pos == 0 or num_total_negs == 0:
        print("Stage-0 skipped: insufficient positive or negative samples.")
        return {"enabled": False, "reason": "insufficient_samples"}

    split_idx = max(1, int(0.8 * len(samples)))
    split_idx = min(split_idx, len(samples) - 1)
    train_stage0 = samples[:split_idx]
    val_stage0 = samples[split_idx:]

    train_ds = LiverAdaptDataset(train_stage0)
    val_ds = LiverAdaptDataset(val_stage0)
    train_dl = DataLoader(train_ds, batch_size=max(1, args.stage0_batch_size), shuffle=True, collate_fn=stage0_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=max(1, args.stage0_batch_size), shuffle=False, collate_fn=stage0_collate_fn)

    unfreeze_info = configure_stage0_trainable(clip, args.stage0_unfreeze_last_n)
    model = LiverAwareAdapter(clip).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.stage0_lr, weight_decay=args.stage0_wd)

    print(
        f"\nStage-0 liver-awareness warm-up: "
        f"pos={num_pos}, neg={num_total_negs} (external_neg={num_external_negs}), "
        f"train={len(train_stage0)}, val={len(val_stage0)}, unfreeze_last_n={args.stage0_unfreeze_last_n}, "
        f"spatial={unfreeze_info['spatial_unfrozen_blocks']}/{unfreeze_info['spatial_total_blocks']}, "
        f"temporal={unfreeze_info['temporal_unfrozen_blocks']}/{unfreeze_info['temporal_total_blocks']}"
    )

    history = []
    for epoch in range(1, args.stage0_epochs + 1):
        model.train()
        epoch_losses = []
        for scan_paths, labels, crop_modes, phases in train_dl:
            images = []
            for path, crop_mode, phase in zip(scan_paths, crop_modes, phases):
                img = nii_img_to_tensor_regression(
                    path,
                    target_depth=args.target_depth,
                    preprocess_cfg=args,
                    phase=phase,
                    crop_mode_override=crop_mode,
                )
                images.append(img)
            images = torch.stack(images, dim=0).to(device)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            text_tokens = tokenize_text(tokenizer, args.stage0_prompt, device, batch_size=images.size(0))
            logits = model(text_tokens, images, device)
            loss = F.binary_cross_entropy_with_logits(logits, labels_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().numpy()))

        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for scan_paths, labels, crop_modes, phases in val_dl:
                images = []
                for path, crop_mode, phase in zip(scan_paths, crop_modes, phases):
                    img = nii_img_to_tensor_regression(
                        path,
                        target_depth=args.target_depth,
                        preprocess_cfg=args,
                        phase=phase,
                        crop_mode_override=crop_mode,
                    )
                    images.append(img)
                images = torch.stack(images, dim=0).to(device)
                text_tokens = tokenize_text(tokenizer, args.stage0_prompt, device, batch_size=images.size(0))
                logits = model(text_tokens, images, device)
                probs = torch.sigmoid(logits)
                val_probs.extend([float(x) for x in probs.detach().cpu().numpy().tolist()])
                val_labels.extend([int(round(x)) for x in labels])

        val_metrics = compute_group_metrics(val_labels, val_probs, threshold=0.5)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_prob_mean": float(np.mean(val_probs)) if val_probs else float("nan"),
            "val_prob_std": float(np.std(val_probs)) if val_probs else float("nan"),
        }
        history.append(row)
        print(
            f"Stage-0 Epoch {epoch}/{args.stage0_epochs} - "
            f"loss={row['loss']:.4f} val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_f1']:.4f} val_auc={row['val_auc']:.4f}"
        )

    pd.DataFrame(history).to_csv(out_dir / "stage0_liver_adapt_metrics.csv", index=False)
    return {
        "enabled": True,
        "reason": "ok",
        "num_pos": num_pos,
        "num_neg": num_total_negs,
        "num_external_neg": num_external_negs,
        "last_val_auc": history[-1]["val_auc"] if history else float("nan"),
        "unfreeze_last_n": unfreeze_info["unfreeze_last_n"],
        "spatial_unfrozen_blocks": unfreeze_info["spatial_unfrozen_blocks"],
        "spatial_total_blocks": unfreeze_info["spatial_total_blocks"],
        "temporal_unfrozen_blocks": unfreeze_info["temporal_unfrozen_blocks"],
        "temporal_total_blocks": unfreeze_info["temporal_total_blocks"],
    }


class Regressor(nn.Module):
    """Multi-task head (group classification + ratio regression) for CT-CLIP."""
    def __init__(self, clip, use_text=True, hidden=256, dropout=0.2):
        super().__init__()
        self.clip = clip
        self.use_text = use_text
        in_dim = 512 + (512 if use_text else 0)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.group_head = nn.Linear(hidden, 1)   # BCEWithLogits
        self.ratio_head = nn.Linear(hidden, 1)   # sigmoid -> [0, 1]

    def forward(self, text_tokens, images, device):
        text_latents, image_latents, _ = self.clip(
            text_tokens, images, device=device, return_latents=True
        )
        if self.use_text:
            feats = torch.cat([image_latents, text_latents], dim=1)
        else:
            feats = image_latents
        hidden = self.backbone(feats)
        group_logit = self.group_head(hidden).squeeze(1)
        ratio_pred = torch.sigmoid(self.ratio_head(hidden)).squeeze(1)
        return group_logit, ratio_pred


class TextOnlyRegressor(nn.Module):
    """Text-only multi-task head for tumor-marker ablation."""
    def __init__(self, clip, hidden=256, dropout=0.2):
        super().__init__()
        self.clip = clip
        self.backbone = nn.Sequential(
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.group_head = nn.Linear(hidden, 1)   # BCEWithLogits
        self.ratio_head = nn.Linear(hidden, 1)   # sigmoid -> [0, 1]

    def forward(self, text_tokens):
        text_embeddings = self.clip.text_transformer(
            text_tokens.input_ids, attention_mask=text_tokens.attention_mask
        )
        enc_text = text_embeddings[0]
        text_embeds = enc_text[:, 0, :]
        text_latents = self.clip.to_text_latent(text_embeds)
        text_latents = F.normalize(text_latents, dim=-1)
        hidden = self.backbone(text_latents)
        group_logit = self.group_head(hidden).squeeze(1)
        ratio_pred = torch.sigmoid(self.ratio_head(hidden)).squeeze(1)
        return group_logit, ratio_pred


def build_samples(
    hcc_root,
    excel_path,
    target_col,
    group_col="坏死比例分组",
    require_scans=True,
    necrosis_mode="group_only",
):
    """Build samples from HCC dataset."""
    df = pd.read_excel(excel_path)
    if necrosis_mode != "group_only" and target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
    has_ratio_target = target_col in df.columns

    by_name = {}
    by_id = {}
    for _, row in df.iterrows():
        if "NAME" in df.columns and not pd.isna(row["NAME"]):
            try:
                by_name[int(row["NAME"])] = row
            except Exception:
                pass
        if "ID" in df.columns and not pd.isna(row["ID"]):
            try:
                by_id[int(row["ID"])] = row
            except Exception:
                pass

    samples = []
    patient_dirs = sorted([p for p in os.listdir(hcc_root) if p.startswith("ID")])

    for pdir in patient_dirs:
        pid, pname = parse_folder_name(pdir)
        row = None
        if pname is not None:
            row = by_name.get(pname)
        if row is None and pid is not None:
            row = by_id.get(pid)
        if row is None:
            continue

        ratio_target_val = safe_float(row[target_col], None) if has_ratio_target else None
        group_target_raw = safe_float(row[group_col], None)
        group_target_val = None
        if group_target_raw is not None:
            group_target_val = 1.0 if group_target_raw >= 0.5 else 0.0
        if necrosis_mode == "group_only":
            # Classification-only mode: use 0/1 group label as target placeholder.
            ratio_target_val = group_target_val
        features = {str(col): row[col] for col in df.columns}
        scans = []
        full_dir = os.path.join(hcc_root, pdir)
        for f in sorted(os.listdir(full_dir)):
            if f.endswith(".nii.gz"):
                scans.append(os.path.join(full_dir, f))
        if require_scans and not scans:
            continue
        samples.append(
            Sample(
                patient_key=pdir,
                features=features,
                ratio_target=ratio_target_val,
                group_target=group_target_val,
                scans=scans,
            )
        )

    norm_targets, scale = normalize_target([s.ratio_target for s in samples])
    for s, t in zip(samples, norm_targets):
        s.ratio_target = t

    # Remove samples with missing labels.
    if necrosis_mode == "group_only":
        samples = [s for s in samples if s.group_target is not None]
    else:
        samples = [s for s in samples if s.ratio_target is not None and s.group_target is not None]
    return samples, scale


def split_samples(samples, train_ratio, train_n, seed):
    """Split samples into train and test sets."""
    if len(samples) < 2:
        raise ValueError("Need at least 2 patients to create train/test split.")
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    if train_n is not None:
        train_n = max(1, min(train_n, len(samples) - 1))
        train_idx = idx[:train_n]
        test_idx = idx[train_n:]
    else:
        if train_ratio is None:
            train_ratio = 0.7
        split = max(1, int(len(samples) * train_ratio))
        split = min(split, len(samples) - 1)
        train_idx = idx[:split]
        test_idx = idx[split:]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def resolve_path_maybe_relative(path_value, out_dir):
    """Resolve path; relative paths are resolved under out_dir."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    return Path(out_dir) / p


def compute_samples_universe_hash(samples):
    """Stable hash of all available patient ids for split consistency checks."""
    keys = sorted(s.patient_key for s in samples)
    text = "\n".join(keys)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_no_split_overlap(train_keys, test_keys):
    """Ensure train/test are disjoint."""
    overlap = sorted(set(train_keys) & set(test_keys))
    if overlap:
        raise ValueError(f"Train/test split overlap detected: {overlap[:10]}")


def ensure_unique_keys(keys, split_name):
    """Ensure a split list has no duplicate patient keys."""
    if len(keys) != len(set(keys)):
        seen = set()
        dup = []
        for k in keys:
            if k in seen:
                dup.append(k)
                if len(dup) >= 10:
                    break
            seen.add(k)
        raise ValueError(f"Duplicate patient keys in {split_name}: {dup}")


def file_sha256(path):
    """Compute SHA256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path):
    """Lightweight file fingerprint for manifest tracking."""
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def split_manifest_payload(args, samples, train_samples, test_samples):
    """Create split manifest payload."""
    train_keys = [s.patient_key for s in train_samples]
    test_keys = [s.patient_key for s in test_samples]
    ensure_unique_keys(train_keys, "train_patients")
    ensure_unique_keys(test_keys, "test_patients")
    ensure_no_split_overlap(train_keys, test_keys)
    return {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hcc_root": str(args.hcc_root),
        "excel": str(args.excel),
        "target_col": args.target_col,
        "group_col": args.group_col,
        "split_policy": {
            "train_n": args.train_n,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
        "sample_universe_hash": compute_samples_universe_hash(samples),
        "sample_universe_size": len(samples),
        "train_patients": train_keys,
        "train_patients_count": len(train_keys),
        "test_patients": test_keys,
        "test_patients_count": len(test_keys),
    }


def save_split_manifest(path, payload):
    """Write split manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path):
    """Load JSON from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def samples_from_patient_keys(samples, train_keys, test_keys):
    """Build split samples from explicit patient key lists."""
    ensure_unique_keys(train_keys, "train_patients")
    ensure_unique_keys(test_keys, "test_patients")
    ensure_no_split_overlap(train_keys, test_keys)
    by_key = {s.patient_key: s for s in samples}
    missing_train = [k for k in train_keys if k not in by_key]
    missing_test = [k for k in test_keys if k not in by_key]
    if missing_train or missing_test:
        raise ValueError(
            f"Split manifest contains patients missing from current dataset. "
            f"missing_train={missing_train[:10]}, missing_test={missing_test[:10]}"
        )
    train_samples = [by_key[k] for k in train_keys]
    test_samples = [by_key[k] for k in test_keys]
    return train_samples, test_samples


def regression_config_signature(args):
    """Configuration signature for strict train/test consistency checks."""
    return {
        "prompt_template": args.prompt_template,
        "scan_handling": args.scan_handling,
        "train_n": args.train_n,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "target_col": args.target_col,
        "group_col": args.group_col,
        "necrosis_mode": args.necrosis_mode,
        "train_mode": args.train_mode,
        "use_text": bool(args.use_text),
        "loss_weight_group": float(args.loss_weight_group),
        "loss_weight_ratio": float(args.loss_weight_ratio),
        "group_threshold": float(args.group_threshold),
        "liver_prior_crop": args.liver_prior_crop,
        "liver_window": bool(args.liver_window),
        "phase_norm": bool(args.phase_norm),
        "enable_stage0_liver_adapt": bool(args.enable_stage0_liver_adapt),
        "stage0_epochs": int(args.stage0_epochs),
        "stage0_lr": float(args.stage0_lr),
        "stage0_batch_size": int(args.stage0_batch_size),
        "stage0_wd": float(args.stage0_wd),
        "stage0_unfreeze_last_n": int(args.stage0_unfreeze_last_n),
        "stage0_negative_root": str(args.stage0_negative_root),
        "stage0_max_negatives": int(args.stage0_max_negatives),
        "stage0_use_pseudo_negatives": bool(args.stage0_use_pseudo_negatives),
        "stage0_prompt": str(args.stage0_prompt),
    }


TEST_IGNORED_CONFIG_KEYS = {
    "enable_stage0_liver_adapt",
    "stage0_epochs",
    "stage0_lr",
    "stage0_batch_size",
    "stage0_wd",
    "stage0_unfreeze_last_n",
    "stage0_negative_root",
    "stage0_max_negatives",
    "stage0_use_pseudo_negatives",
    "stage0_prompt",
}


def config_for_test_compare(cfg):
    """Drop training-only keys when validating test-time config compatibility."""
    out = dict(cfg)
    for k in TEST_IGNORED_CONFIG_KEYS:
        out.pop(k, None)
    return out


def config_mismatch_messages(current_cfg, expected_cfg):
    """Return list of mismatch messages between two config signatures."""
    mismatches = []
    for k, expected in expected_cfg.items():
        current = current_cfg.get(k, None)
        if current != expected:
            mismatches.append(f"{k}: expected={expected} current={current}")
    return mismatches


def collate_fn(batch, device):
    """Collate function for DataLoader."""
    patient_keys, feature_rows, ratio_targets, group_targets, scan_lists = zip(*batch)
    ratio_targets = torch.tensor(ratio_targets, dtype=torch.float32, device=device)
    group_targets = torch.tensor(group_targets, dtype=torch.float32, device=device)
    return patient_keys, feature_rows, ratio_targets, group_targets, scan_lists


def predict_patient(model, tokenizer, feature_row, scans, args, device):
    """Predict one patient by aggregating selected scan predictions."""
    if args.prompt_template == "tumor_markers_text_only":
        text = build_prompt_text(
            feature_row,
            prompt_template=args.prompt_template,
            target_col=args.target_col,
            phase=None,
        )
        text_tokens = tokenize_text(tokenizer, text, device)
        group_logit, ratio_pred = model(text_tokens)
        return group_logit.squeeze(0), ratio_pred.squeeze(0)

    selected_scans = select_scans_for_patient(scans, args.prompt_template)
    if not selected_scans:
        raise ValueError("No scan selected for patient.")

    per_scan_group_logits = []
    per_scan_ratio_preds = []
    shared_tokens = None
    if args.scan_handling == "ignore":
        text = build_prompt_text(
            feature_row,
            prompt_template=args.prompt_template,
            target_col=args.target_col,
            phase=None,
        )
        shared_tokens = tokenize_text(tokenizer, text, device)

    for scan in selected_scans:
        if shared_tokens is None:
            text = build_prompt_text(
                feature_row,
                prompt_template=args.prompt_template,
                target_col=args.target_col,
                phase=scan_phase(scan),
            )
            text_tokens = tokenize_text(tokenizer, text, device)
        else:
            text_tokens = shared_tokens

        img = nii_img_to_tensor_regression(
            scan,
            target_depth=args.target_depth,
            preprocess_cfg=args,
            phase=scan_phase(scan),
        ).unsqueeze(0).to(device)
        group_logit, ratio_pred = model(text_tokens, img, device)
        per_scan_group_logits.append(group_logit.squeeze(0))
        per_scan_ratio_preds.append(ratio_pred.squeeze(0))

    return torch.stack(per_scan_group_logits).mean(), torch.stack(per_scan_ratio_preds).mean()


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "pearson": float("nan")}
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Pearson
    if len(y_true) > 1:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": corr}


def compute_group_metrics(y_true, y_prob, threshold=0.5):
    """Compute binary classification metrics from probabilities."""
    if len(y_true) == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    acc = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    auc = float("nan")
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(y_prob)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_prob) + 1, dtype=np.float64)
        unique_vals, inv, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
        for idx, count in enumerate(counts):
            if count > 1:
                tie_inds = np.where(inv == idx)[0]
                ranks[tie_inds] = np.mean(ranks[tie_inds])
        sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": float(auc)}


def apply_group_consistency(group_pred, ratio_pred):
    """Apply export-time consistency: group=1 -> ratio=1.0, else cap at 0.99."""
    if int(group_pred) == 1:
        return 1.0
    return min(float(ratio_pred), 0.99)


def evaluate_multitask(model, tokenizer, dataloader, args, device):
    """Evaluate multitask model and return metrics + prediction rows."""
    model.eval()
    group_only = getattr(args, "necrosis_mode", "group_only") == "group_only"
    ratio_true, ratio_pred_raw = [], []
    group_true, group_prob = [], []
    rows = []

    with torch.no_grad():
        for patient_keys, feature_rows, ratio_targets, group_targets, scan_lists in dataloader:
            batch_group_logits, batch_ratio_preds = [], []
            for feature_row, scans in zip(feature_rows, scan_lists):
                group_logit, ratio_pred = predict_patient(model, tokenizer, feature_row, scans, args, device)
                batch_group_logits.append(group_logit)
                batch_ratio_preds.append(ratio_pred)
            batch_group_logits = torch.stack(batch_group_logits)
            batch_ratio_preds = torch.stack(batch_ratio_preds)
            batch_group_probs = torch.sigmoid(batch_group_logits)

            for k, rt, gt, gp, rp in zip(patient_keys, ratio_targets, group_targets, batch_group_probs, batch_ratio_preds):
                rt_v = float(rt.detach().cpu().numpy())
                gt_v = int(round(float(gt.detach().cpu().numpy())))
                gp_v = float(gp.detach().cpu().numpy())
                rp_v = float(rp.detach().cpu().numpy())
                gp_raw = 1 if gp_v >= args.group_threshold else 0
                rp_final = apply_group_consistency(gp_raw, rp_v)

                if not group_only:
                    ratio_true.append(rt_v)
                    ratio_pred_raw.append(rp_v)
                group_true.append(gt_v)
                group_prob.append(gp_v)
                if group_only:
                    rows.append(
                        {
                            "patient": k,
                            "group_target": gt_v,
                            "group_prob": gp_v,
                            "group_pred": gp_raw,
                        }
                    )
                else:
                    rows.append(
                        {
                            "patient": k,
                            "ratio_target": rt_v,
                            "ratio_pred": rp_final,
                            "group_target": gt_v,
                            "group_prob": gp_v,
                            "group_pred": gp_raw,
                        }
                    )

    if group_only:
        ratio_metrics = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "pearson": float("nan")}
    else:
        ratio_metrics = compute_metrics(ratio_true, ratio_pred_raw)
    group_metrics = compute_group_metrics(group_true, group_prob, threshold=args.group_threshold)
    return ratio_metrics, group_metrics, rows


def run_regression(args):
    """Run regression training and testing."""
    from transformer_maskgit import CTViT

    if not hasattr(args, "prompt_template"):
        args.prompt_template = "arterial_only"
    if not hasattr(args, "scan_handling"):
        args.scan_handling = "distinguish"
    if not hasattr(args, "auto_out_subdir"):
        args.auto_out_subdir = True
    if not hasattr(args, "run_name"):
        args.run_name = None
    if not hasattr(args, "train_mode"):
        args.train_mode = "lipro"
    if not hasattr(args, "official_finetune"):
        args.official_finetune = False
    if not hasattr(args, "wd"):
        args.wd = 0.1
    if not hasattr(args, "warmup_length"):
        args.warmup_length = 10
    if not hasattr(args, "use_text"):
        args.use_text = True
    if not hasattr(args, "group_col"):
        args.group_col = "坏死比例分组"
    if not hasattr(args, "necrosis_mode"):
        args.necrosis_mode = "group_only"
    if not hasattr(args, "loss_weight_group"):
        args.loss_weight_group = 0.8
    if not hasattr(args, "loss_weight_ratio"):
        args.loss_weight_ratio = 0.2
    if not hasattr(args, "group_threshold"):
        args.group_threshold = 0.5
    if not hasattr(args, "liver_prior_crop"):
        args.liver_prior_crop = "right_upper_abdomen"
    if not hasattr(args, "liver_window"):
        args.liver_window = True
    if not hasattr(args, "phase_norm"):
        args.phase_norm = True
    if not hasattr(args, "enable_stage0_liver_adapt"):
        args.enable_stage0_liver_adapt = False
    if not hasattr(args, "stage0_epochs"):
        args.stage0_epochs = 5
    if not hasattr(args, "stage0_lr"):
        args.stage0_lr = 5e-5
    if not hasattr(args, "stage0_batch_size"):
        args.stage0_batch_size = 2
    if not hasattr(args, "stage0_wd"):
        args.stage0_wd = 1e-4
    if not hasattr(args, "stage0_unfreeze_last_n"):
        args.stage0_unfreeze_last_n = 1
    if not hasattr(args, "stage0_unfreeze_scope"):
        args.stage0_unfreeze_scope = None
    if not hasattr(args, "stage0_negative_root"):
        args.stage0_negative_root = str(Path(args.hcc_root).parents[1] / "valid")
    if not hasattr(args, "stage0_max_negatives"):
        args.stage0_max_negatives = 64
    if not hasattr(args, "stage0_use_pseudo_negatives"):
        args.stage0_use_pseudo_negatives = True
    if not hasattr(args, "stage0_prompt"):
        args.stage0_prompt = "This is a preoperative abdominal contrast-enhanced CT volume for organ localization."
    if not hasattr(args, "split_file"):
        args.split_file = None

    if args.disable_proxy:
        apply_proxy_env()

    args.scan_handling = canonical_scan_handling(args.scan_handling)
    if args.prompt_template == "tumor_markers_text_only":
        args.scan_handling = "text_only"
        args.use_text = True
        args.enable_stage0_liver_adapt = False
    if args.necrosis_mode == "group_only":
        if args.loss_weight_group != 1.0 or args.loss_weight_ratio != 0.0:
            print(
                "Info: necrosis_mode=group_only; overriding loss weights to "
                "group=1.0, ratio=0.0."
            )
        args.loss_weight_group = 1.0
        args.loss_weight_ratio = 0.0
    if args.stage0_unfreeze_scope is not None:
        legacy_map = {
            "visual_projection": 0,
            "visual_projection_plus_last": 1,
            "all_visual": -1,
        }
        args.stage0_unfreeze_last_n = legacy_map[args.stage0_unfreeze_scope]
    # Backward compatibility: --official-finetune promotes mode to vocabfine.
    if args.official_finetune and args.train_mode == "lipro":
        args.train_mode = "vocabfine"
    # Canonical mode flags
    if args.train_mode == "vocabfine":
        args.official_finetune = True
        args.freeze_clip = False
    else:
        args.official_finetune = False
        args.freeze_clip = True

    out_dir, config_subdir = resolve_regression_out_dir(args)
    set_seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task: regression")
    print(f"HCC root: {args.hcc_root}")
    print(f"Excel: {args.excel}")
    print(f"Target column: {args.target_col}")
    print(f"Group column: {args.group_col}")
    print(f"Necrosis mode: {args.necrosis_mode}")
    print(f"Output root: {args.out_dir}")
    if config_subdir is not None:
        print(f"Output subdir: {config_subdir}")
    print(f"Output dir: {out_dir}")
    print(f"Stage: {args.stage}")
    print(f"Patient unit: patient (two sequences)")
    print(f"Scan handling: {args.scan_handling}")
    print(f"Prompt template: {args.prompt_template}")
    print(f"Template desc: {PROMPT_TEMPLATE_DESCRIPTIONS[args.prompt_template]}")
    print(f"Liver prior crop: {args.liver_prior_crop}")
    print(f"Liver window: {args.liver_window}")
    print(f"Phase norm: {args.phase_norm}")
    print(f"Loss weights (group/ratio): {args.loss_weight_group}/{args.loss_weight_ratio}")
    print(f"Group threshold: {args.group_threshold}")
    print(f"Train mode: {args.train_mode}")
    print(f"Official fine-tune mode: {args.official_finetune}")
    print(f"Stage-0 liver adapt: {args.enable_stage0_liver_adapt}")
    if args.enable_stage0_liver_adapt:
        print(
            f"Stage-0 config: epochs={args.stage0_epochs}, lr={args.stage0_lr}, "
            f"batch={args.stage0_batch_size}, unfreeze_last_n={args.stage0_unfreeze_last_n}, "
            f"neg_root={args.stage0_negative_root}"
        )

    samples, scale = build_samples(
        args.hcc_root,
        args.excel,
        args.target_col,
        group_col=args.group_col,
        require_scans=(args.prompt_template != "tumor_markers_text_only"),
        necrosis_mode=args.necrosis_mode,
    )
    detected_marker_cols = []
    if samples:
        detected_marker_cols = select_tumor_marker_columns(samples[0].features, args.target_col)
    if args.prompt_template == "tumor_markers_text_only":
        print(f"Marker columns: {detected_marker_cols}")
        if not detected_marker_cols:
            print("Warning: no tumor-marker column detected in Excel; prompts will contain missing markers only.")
        print("CT usage: disabled (text-only ablation)")

    split_manifest = None
    split_manifest_path = None
    split_manifest_hash = None

    if args.stage == "test":
        if not args.load_model:
            raise ValueError("Test stage requires --load-model.")
        if not args.split_file:
            raise ValueError("Test stage requires --split-file to ensure fixed train/test split.")
        split_manifest_path = resolve_path_maybe_relative(args.split_file, out_dir)
        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
        split_manifest = load_json(split_manifest_path)
        for required_key in ("train_patients", "test_patients", "sample_universe_hash"):
            if required_key not in split_manifest:
                raise ValueError(f"Invalid split manifest: missing key `{required_key}`")
        expected_universe_hash = split_manifest.get("sample_universe_hash", "")
        current_universe_hash = compute_samples_universe_hash(samples)
        if expected_universe_hash != current_universe_hash:
            raise ValueError(
                "Dataset universe changed vs split manifest. "
                f"expected={expected_universe_hash}, current={current_universe_hash}"
            )
        expected_universe_size = int(split_manifest.get("sample_universe_size", len(samples)))
        if expected_universe_size != len(samples):
            raise ValueError(
                "Dataset size changed vs split manifest. "
                f"expected={expected_universe_size}, current={len(samples)}"
            )
        train_keys = split_manifest.get("train_patients", [])
        test_keys = split_manifest.get("test_patients", [])
        expected_train_count = split_manifest.get("train_patients_count", None)
        expected_test_count = split_manifest.get("test_patients_count", None)
        if expected_train_count is not None and int(expected_train_count) != len(train_keys):
            raise ValueError(
                "Split manifest train count mismatch: "
                f"declared={expected_train_count}, actual_list={len(train_keys)}"
            )
        if expected_test_count is not None and int(expected_test_count) != len(test_keys):
            raise ValueError(
                "Split manifest test count mismatch: "
                f"declared={expected_test_count}, actual_list={len(test_keys)}"
            )
        train_samples, test_samples = samples_from_patient_keys(samples, train_keys, test_keys)
        split_manifest_hash = file_sha256(split_manifest_path)
    else:
        train_samples, test_samples = split_samples(samples, args.train_ratio, args.train_n, args.seed)
        split_manifest_path = resolve_path_maybe_relative(args.split_file, out_dir) if args.split_file else (out_dir / "split_manifest.json")
        split_manifest = split_manifest_payload(args, samples, train_samples, test_samples)
        save_split_manifest(split_manifest_path, split_manifest)
        split_manifest_hash = file_sha256(split_manifest_path)
        args.split_file = str(split_manifest_path)

    print(f"Total samples: {len(samples)}, Train: {len(train_samples)}, Test: {len(test_samples)}")
    print(f"Split file: {split_manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=args.temporal_patch_size,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    )

    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )

    pt = torch.load(args.checkpoint, map_location="cpu")
    clip.load_state_dict(pt, strict=False)
    clip.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage0_meta = {"enabled": False, "reason": "disabled"}
    if args.enable_stage0_liver_adapt and args.stage in ("train", "both") and args.prompt_template != "tumor_markers_text_only":
        stage0_meta = run_stage0_liver_adapt(clip, tokenizer, train_samples, args, device, out_dir)

    if args.official_finetune:
        # Match official CT-CLIP fine-tuning style: train end-to-end.
        for p in clip.parameters():
            p.requires_grad = True
    elif args.freeze_clip:
        for p in clip.parameters():
            p.requires_grad = False
        clip.eval()

    if args.prompt_template == "tumor_markers_text_only":
        model = TextOnlyRegressor(clip).to(device)
    else:
        model = Regressor(clip, use_text=args.use_text).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)

    train_ds = HCCDataset(train_samples)
    test_ds = HCCDataset(test_samples)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, device),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, device),
    )
    total_train_steps = max(1, args.epochs * max(1, len(train_dl)))
    warmup_steps = min(max(0, args.warmup_length), max(0, total_train_steps - 1))
    scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_train_steps)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model is None:
        args.save_model = str(out_dir / "regressor.pt")
    elif not os.path.isabs(args.save_model):
        args.save_model = str(out_dir / args.save_model)

    if args.batch_size != 1:
        raise ValueError("This script currently requires --batch-size 1 to keep token/image alignment.")

    # Persist split for reproducibility
    pd.DataFrame({"patient": [s.patient_key for s in train_samples]}).to_csv(out_dir / "train_patients.csv", index=False)
    pd.DataFrame({"patient": [s.patient_key for s in test_samples]}).to_csv(out_dir / "test_patients.csv", index=False)

    if args.load_model:
        if not os.path.isabs(args.load_model):
            args.load_model = str(out_dir / args.load_model)

        if args.stage == "test":
            ckpt_manifest_path = Path(args.load_model).parent / "checkpoint_manifest.json"
            if not ckpt_manifest_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint manifest not found: {ckpt_manifest_path}. "
                    "Cannot run strict consistency check."
                )
            ckpt_manifest = load_json(ckpt_manifest_path)
            expected_cfg = ckpt_manifest.get("config_signature", {})
            current_cfg = regression_config_signature(args)
            # Test-time does not execute Stage-0; ignore Stage-0 training knobs in strict match.
            expected_cfg = config_for_test_compare(expected_cfg)
            current_cfg = config_for_test_compare(current_cfg)
            mismatches = config_mismatch_messages(current_cfg, expected_cfg)
            if mismatches:
                raise ValueError(
                    "Config mismatch between checkpoint and current test run:\n- " + "\n- ".join(mismatches)
                )

            expected_split_hash = ckpt_manifest.get("split_manifest_sha256", "")
            if expected_split_hash and split_manifest_hash and expected_split_hash != split_manifest_hash:
                raise ValueError(
                    "Split manifest mismatch with checkpoint manifest: "
                    f"expected={expected_split_hash}, current={split_manifest_hash}"
                )

            expected_universe_hash = ckpt_manifest.get("sample_universe_hash", "")
            current_universe_hash = compute_samples_universe_hash(samples)
            if expected_universe_hash and expected_universe_hash != current_universe_hash:
                raise ValueError(
                    "Dataset universe mismatch with checkpoint manifest: "
                    f"expected={expected_universe_hash}, current={current_universe_hash}"
                )

            print(f"Checkpoint manifest verified: {ckpt_manifest_path}")

        model.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")

    if args.stage in ("train", "both"):
        # Training logs saved per configuration directory
        train_config_path = out_dir / "train_config.json"
        train_lr_log_path = out_dir / "train_lr_log.csv"
        train_epoch_log_path = out_dir / "train_epoch_log.csv"
        train_cfg_payload = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "output_dir": str(out_dir),
            "save_model": str(args.save_model),
            "checkpoint": str(args.checkpoint),
            "split_file": str(split_manifest_path) if split_manifest_path else "",
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "total_train_steps": int(total_train_steps),
            "warmup_steps": int(warmup_steps),
            "config_signature": regression_config_signature(args),
            "args": dict(sorted(vars(args).items())),
        }
        with open(train_config_path, "w", encoding="utf-8") as f:
            json.dump(train_cfg_payload, f, ensure_ascii=False, indent=2)
        print(f"Training config log saved to {train_config_path}")

        print("\nStarting training...")
        global_step = 0
        lr_rows = []
        epoch_rows = []
        rows = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, group_losses, ratio_losses = [], [], []
            group_only = args.necrosis_mode == "group_only"
            for _, feature_rows, ratio_targets, group_targets, scan_lists in train_dl:
                scheduler(global_step)
                current_lr = float(optimizer.param_groups[0]["lr"])
                lr_rows.append(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "lr": current_lr,
                    }
                )
                group_logits, ratio_preds = [], []
                for feature_row, scans in zip(feature_rows, scan_lists):
                    group_logit, ratio_pred = predict_patient(model, tokenizer, feature_row, scans, args, device)
                    group_logits.append(group_logit)
                    ratio_preds.append(ratio_pred)
                group_logits = torch.stack(group_logits)
                ratio_preds = torch.stack(ratio_preds)
                loss_group = F.binary_cross_entropy_with_logits(group_logits, group_targets)
                if group_only:
                    loss_ratio = torch.zeros((), dtype=loss_group.dtype, device=loss_group.device)
                    loss = loss_group
                else:
                    loss_ratio = F.mse_loss(ratio_preds, ratio_targets)
                    loss = args.loss_weight_group * loss_group + args.loss_weight_ratio * loss_ratio
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu().numpy()))
                group_losses.append(float(loss_group.detach().cpu().numpy()))
                ratio_losses.append(float(loss_ratio.detach().cpu().numpy()))
                global_step += 1

            # Evaluate on test set
            ratio_metrics, group_metrics, rows = evaluate_multitask(model, tokenizer, test_dl, args, device)
            epoch_row = {
                "epoch": int(epoch),
                "global_step_end": int(global_step),
                "lr_end": float(optimizer.param_groups[0]["lr"]),
                "loss": float(np.mean(losses)),
                "loss_group": float(np.mean(group_losses)),
                "loss_ratio": float(np.mean(ratio_losses)),
                "ratio_mae": float(ratio_metrics["mae"]),
                "ratio_rmse": float(ratio_metrics["rmse"]),
                "ratio_r2": float(ratio_metrics["r2"]),
                "ratio_pearson": float(ratio_metrics["pearson"]),
                "group_acc": float(group_metrics["acc"]),
                "group_precision": float(group_metrics["precision"]),
                "group_recall": float(group_metrics["recall"]),
                "group_f1": float(group_metrics["f1"]),
                "group_auc": float(group_metrics["auc"]),
            }
            epoch_rows.append(epoch_row)
            if group_only:
                print(
                    f"Epoch {epoch}/{args.epochs} - loss={np.mean(losses):.4f} "
                    f"(group={np.mean(group_losses):.4f}) "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"group_acc={group_metrics['acc']:.4f} group_f1={group_metrics['f1']:.4f} "
                    f"group_auc={group_metrics['auc']:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}/{args.epochs} - loss={np.mean(losses):.4f} "
                    f"(group={np.mean(group_losses):.4f}, ratio={np.mean(ratio_losses):.4f}) "
                    f"ratio_mae={ratio_metrics['mae']:.4f} ratio_rmse={ratio_metrics['rmse']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e} "
                    f"group_acc={group_metrics['acc']:.4f} group_f1={group_metrics['f1']:.4f}"
                )

        if not rows:
            # Allow edge-case runs like --epochs 0 while still exporting predictions.
            _, _, rows = evaluate_multitask(model, tokenizer, test_dl, args, device)

        pd.DataFrame(lr_rows).to_csv(train_lr_log_path, index=False)
        pd.DataFrame(epoch_rows).to_csv(train_epoch_log_path, index=False)
        print(f"Training LR log saved to {train_lr_log_path}")
        print(f"Training epoch log saved to {train_epoch_log_path}")

        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")
        ckpt_manifest = {
            "version": 1,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model_path": str(Path(args.save_model).resolve()),
            "base_checkpoint": file_fingerprint(args.checkpoint),
            "config_signature": regression_config_signature(args),
            "split_manifest_path": str(split_manifest_path.resolve()) if split_manifest_path else "",
            "split_manifest_sha256": split_manifest_hash or "",
            "sample_universe_hash": compute_samples_universe_hash(samples),
            "train_patients_count": len(train_samples),
            "test_patients_count": len(test_samples),
        }
        ckpt_manifest_path = Path(args.save_model).parent / "checkpoint_manifest.json"
        with open(ckpt_manifest_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_manifest, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint manifest saved to {ckpt_manifest_path}")
        pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    if args.stage == "test":
        print("\nRunning test-only evaluation...")
        ratio_metrics, group_metrics, rows = evaluate_multitask(model, tokenizer, test_dl, args, device)
        if args.necrosis_mode == "group_only":
            print(
                f"Test results: group_acc={group_metrics['acc']:.4f} "
                f"group_precision={group_metrics['precision']:.4f} "
                f"group_recall={group_metrics['recall']:.4f} "
                f"group_f1={group_metrics['f1']:.4f} "
                f"group_auc={group_metrics['auc']:.4f}"
            )
        else:
            print(
                f"Test results: ratio_mae={ratio_metrics['mae']:.4f} ratio_rmse={ratio_metrics['rmse']:.4f} "
                f"ratio_r2={ratio_metrics['r2']:.4f} ratio_pearson={ratio_metrics['pearson']:.4f} "
                f"group_acc={group_metrics['acc']:.4f} group_f1={group_metrics['f1']:.4f} "
                f"group_auc={group_metrics['auc']:.4f}"
            )
        pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)

    # Save run metadata
    meta = {
        "target_col": args.target_col,
        "group_col": args.group_col,
        "scale": scale,
        "output_root": str(args.out_dir),
        "output_subdir": config_subdir,
        "output_dir": str(out_dir),
        "split_file": str(split_manifest_path) if split_manifest_path else "",
        "split_manifest_sha256": split_manifest_hash or "",
        "scan_handling": args.scan_handling,
        "phase_convention": "1.nii.gz=arterial;2.nii.gz=portal_venous",
        "patient_unit": "patient",
        "prompt_template": args.prompt_template,
        "image_used": args.prompt_template != "tumor_markers_text_only",
        "marker_columns": "|".join(detected_marker_cols) if args.prompt_template == "tumor_markers_text_only" else "",
        "train_mode": args.train_mode,
        "necrosis_mode": args.necrosis_mode,
        "train_n": args.train_n,
        "train_ratio": args.train_ratio,
        "use_text": args.use_text,
        "loss_weight_group": args.loss_weight_group,
        "loss_weight_ratio": args.loss_weight_ratio,
        "group_threshold": args.group_threshold,
        "consistency_rule": (
            "classification_only: export group_target/group_prob/group_pred only"
            if args.necrosis_mode == "group_only"
            else "export_only: group=1->ratio=1.0; group=0->ratio=min(raw,0.99)"
        ),
        "liver_prior_crop": args.liver_prior_crop,
        "liver_window": args.liver_window,
        "phase_norm": args.phase_norm,
        "stage0_liver_adapt_enabled": bool(stage0_meta.get("enabled", False)),
        "stage0_liver_adapt_reason": stage0_meta.get("reason", ""),
        "stage0_last_val_auc": stage0_meta.get("last_val_auc", float("nan")),
        "stage0_num_pos": stage0_meta.get("num_pos", 0),
        "stage0_num_neg": stage0_meta.get("num_neg", 0),
        "stage0_num_external_neg": stage0_meta.get("num_external_neg", 0),
        "stage0_epochs": args.stage0_epochs,
        "stage0_lr": args.stage0_lr,
        "stage0_batch_size": args.stage0_batch_size,
        "stage0_unfreeze_last_n": args.stage0_unfreeze_last_n,
        "stage0_unfreeze_scope_legacy": args.stage0_unfreeze_scope,
        "stage0_spatial_unfrozen_blocks": stage0_meta.get("spatial_unfrozen_blocks", 0),
        "stage0_spatial_total_blocks": stage0_meta.get("spatial_total_blocks", 0),
        "stage0_temporal_unfrozen_blocks": stage0_meta.get("temporal_unfrozen_blocks", 0),
        "stage0_temporal_total_blocks": stage0_meta.get("temporal_total_blocks", 0),
        "stage0_negative_root": args.stage0_negative_root,
        "stage0_use_pseudo_negatives": args.stage0_use_pseudo_negatives,
        "stage0_prompt": args.stage0_prompt,
        "lr": args.lr,
        "epochs": args.epochs,
        "freeze_clip": args.freeze_clip,
        "official_finetune": args.official_finetune,
        "wd": args.wd,
        "warmup_length": args.warmup_length,
        "save_model": args.save_model,
        "load_model": args.load_model,
        "stage": args.stage,
    }
    pd.DataFrame([meta]).to_csv(out_dir / "run_meta.csv", index=False)
    print(f"\nResults saved to {out_dir}")
    print("Regression complete!")


class CTClipInference(nn.Module):
    def __init__(
        self,
        CTClip: CTCLIP,
        *,
        data_folder: "external_valid",
        reports_file: "data_reports.xslx",
        meta_file: "meta_data.csv",
        results_folder = './results',
        labels = "labels.csv",
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **accelerate_kwargs)
        self.CTClip = CTClip
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
        self.results_folder = results_folder
        self.register_buffer('steps', torch.Tensor([0]))

        # Load the pre-trained weights
        self.ds = CTReportDatasetinfer(data_folder=data_folder, reports_file=reports_file, meta_file=meta_file, labels=labels)

        # Split dataset into train and validation sets
        self.dl = DataLoader(
            self.ds,
            num_workers=6,
            batch_size=1,
            shuffle = True,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.device = self.accelerator.device
        self.CTClip.to(self.device)

        (
 			self.dl_iter,
            self.CTClip,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.CTClip,
        )

        self.result_folder_txt = self.results_folder
        self.results_folder = Path(results_folder)

        self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def infer(self, log_fn=noop):
        device = self.device

        steps = int(self.steps.item())

        # logs
        logs = {}

        with torch.no_grad():

            models_to_evaluate = ((self.CTClip, str(steps)),)

            for model, filename in models_to_evaluate:
                model.eval()
                predictedall=[]
                realall=[]

                accession_names=[]
                pathologies = ['Medical material','Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion','Coronary artery wall calcification', 'Hiatal hernia','Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule','Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern','Peribronchial thickening', 'Consolidation', 'Bronchiectasis','Interlobular septal thickening']
                for i in tqdm.tqdm(range(len(self.ds))):
                    valid_data, text, onehotlabels, acc_name = next(self.dl_iter)

                    plotdir = self.result_folder_txt
                    Path(plotdir).mkdir(parents=True, exist_ok=True)

                    predictedlabels=[]

                    for pathology in pathologies:
                        text = [f"{pathology} is present.", f"{pathology} is not present."]
                        text_tokens=self.tokenizer(
                                        text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

                        output = model(text_tokens, valid_data.cuda(),  device=device)

                        output = apply_softmax(output)

                        append_out=output.detach().cpu().numpy()
                        predictedlabels.append(append_out[0])

                    predictedall.append(predictedlabels)
                    realall.append(onehotlabels.detach().cpu().numpy()[0])
                    accession_names.append(acc_name[0])

                realall=np.array(realall)
                predictedall=np.array(predictedall)

                np.savez(f"{plotdir}labels_weights.npz", data=realall)
                np.savez(f"{plotdir}predicted_weights.npz", data=predictedall)
                with open(f"{plotdir}accessions.txt", "w") as file:
                    for item in accession_names:
                        file.write(item + "\n")


                dfs=evaluate_internal(predictedall,realall,pathologies, plotdir)

                writer = pd.ExcelWriter(f'{plotdir}aurocs.xlsx', engine='xlsxwriter')

                dfs.to_excel(writer, sheet_name='Sheet1', index=False)

                writer.close()

        self.steps += 1

        log_fn(logs)

        self.accelerator.print('Inference complete')
