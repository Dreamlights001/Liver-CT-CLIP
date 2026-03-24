"""
CT-CLIP Unified Inference Script

This script provides a unified interface for:
1. Zero-shot classification (original CT-CLIP task)
2. HCC necrosis regression testing (test-only for safety)

Usage:
  # Zero-shot classification on chest CT
  python run_zero_shot.py --task classify --dataset-type chest

  # Zero-shot classification on HCC data
  python run_zero_shot.py --task classify --dataset-type hcc

  # HCC necrosis regression - Test only (load pretrained model)
  python run_zero_shot.py --task regression --stage test \\
      --load-model /path/to/output/regressor.pt \\
      --out-dir /path/to/output

Note:
  - Regression training is intentionally moved to `ct_lipro_train.py --task regression`
  - Regression test requires `--split-file` produced by training to enforce fixed split
"""

import argparse
import os
from pathlib import Path
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
import torch
from zero_shot import CTClipInference, run_regression


def parse_args():
    parser = argparse.ArgumentParser(description="CT-CLIP: Unified Zero-Shot Inference and Regression")
    repo_root = Path(__file__).resolve().parents[1]

    # Task type
    parser.add_argument(
        "--task",
        type=str,
        default="classify",
        choices=["classify", "regression"],
        help="Task type: classify (original zero-shot) or regression (HCC necrosis rate)"
    )

    # Dataset type (for classification)
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="chest",
        choices=["chest", "hcc"],
        help="Dataset type: chest (original) or hcc (liver HCC)"
    )

    # Common paths
    parser.add_argument("--checkpoint", type=str, default=str(repo_root / "ckpt/CT-CLIP_v2.pt"),
                        help="Path to CT-CLIP checkpoint")

    # Classification paths
    parser.add_argument("--data-folder", type=str, default=None,
                        help="Data folder path (auto-set based on dataset-type if not specified)")
    parser.add_argument("--reports-file", type=str, default=None,
                        help="Reports CSV file path")
    parser.add_argument("--meta-file", type=str, default=None,
                        help="Metadata CSV file path")
    parser.add_argument("--labels-file", type=str, default=None,
                        help="Labels CSV file path")
    parser.add_argument("--results-folder", type=str, default=None,
                        help="Results output folder")

    # Regression paths
    parser.add_argument("--hcc-root", default=str(repo_root / "datasets/dataset/HCC"),
                        help="HCC dataset root directory")
    parser.add_argument("--excel", default=str(repo_root / "datasets/dataset/HCC/HCC预实验.xlsx"),
                        help="Excel file with clinical data")
    parser.add_argument("--target-col", default="坏死比例",
                        help="Target column for regression")
    parser.add_argument("--group-col", default="坏死比例分组",
                        help="Group label column for binary task (0/1)")
    parser.add_argument("--out-dir", default=str(repo_root / "inference_hcc_regression"),
                        help="Output directory for regression results")
    parser.add_argument(
        "--auto-out-subdir",
        action="store_true",
        default=True,
        help="Automatically create config-specific subdirectory under --out-dir (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-out-subdir",
        dest="auto_out_subdir",
        action="store_false",
        help="Disable config-specific subdirectory and write directly into --out-dir",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional custom subdirectory name when --auto-out-subdir is enabled",
    )

    # Regression parameters
    parser.add_argument(
        "--scan-handling",
        choices=["distinguish", "ignore", "separate", "average"],
        default="distinguish",
        help=(
            "How to use two sequences at patient level: "
            "distinguish (phase-aware prompts), ignore (phase-agnostic prompts). "
            "Legacy aliases: separate->distinguish, average->ignore."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        choices=["arterial_only", "arterial_portal", "all_features", "tumor_markers_text_only"],
        default="arterial_only",
        help=(
            "Prompt template for Excel auxiliary features: "
            "arterial_only (Template-1, arterial CT + age/sex/BMI only), "
            "arterial_portal (Template-2, dual-phase CT + age/sex/BMI only), "
            "all_features (Template-3, dual-phase CT + all non-target clinical fields), "
            "tumor_markers_text_only (Template-4, no CT image; AFP/PIVKA text only)."
        ),
    )
    parser.add_argument("--use-text", action="store_true", default=True,
                        help="Use text features from Excel in regression")
    parser.add_argument("--no-text", dest="use_text", action="store_false",
                        help="Do not use text features")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Training ratio (used only if train-n is not specified)")
    parser.add_argument("--train-n", type=int, default=4,
                        help="Number of training samples (default: 4-shot)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--loss-weight-group", type=float, default=0.8,
                        help="Loss weight for group classification (BCE)")
    parser.add_argument("--loss-weight-ratio", type=float, default=0.2,
                        help="Loss weight for ratio regression (MSE)")
    parser.add_argument("--group-threshold", type=float, default=0.5,
                        help="Probability threshold for group prediction")
    parser.add_argument("--wd", type=float, default=0.1,
                        help="Weight decay (official fine-tuning style)")
    parser.add_argument("--warmup-length", type=int, default=10,
                        help="Warmup steps for cosine scheduler")
    parser.add_argument(
        "--train-mode",
        type=str,
        default="lipro",
        choices=["lipro", "vocabfine"],
        help="Training mode: lipro (freeze CLIP, train regression head) or vocabfine (end-to-end fine-tuning)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--freeze-clip", action="store_true", default=False,
                        help="(Legacy) Freeze CT-CLIP weights during regression training")
    parser.add_argument("--official-finetune", action="store_true", default=False,
                        help="(Legacy) Alias to switch into vocabfine mode")
    parser.add_argument("--no-official-finetune", dest="official_finetune", action="store_false",
                        help="(Legacy) Keep lipro mode")
    parser.add_argument("--disable-proxy", action="store_true", default=True,
                        help="Disable socks proxy to avoid HuggingFace errors")
    parser.add_argument("--temporal-patch-size", type=int, default=10,
                        help="Temporal patch size for image encoder")
    parser.add_argument("--target-depth", type=int, default=240,
                        help="Target depth for volume preprocessing")
    parser.add_argument(
        "--liver-prior-crop",
        type=str,
        default="right_upper_abdomen",
        choices=["none", "right_upper_abdomen"],
        help="Non-segmentation organ prior crop mode for CT input",
    )
    parser.add_argument(
        "--no-liver-prior-crop",
        dest="liver_prior_crop",
        action="store_const",
        const="none",
        help="Disable liver prior crop",
    )
    parser.add_argument(
        "--liver-window",
        action="store_true",
        default=True,
        help="Enable liver window fusion preprocessing",
    )
    parser.add_argument(
        "--no-liver-window",
        dest="liver_window",
        action="store_false",
        help="Disable liver window fusion preprocessing",
    )
    parser.add_argument(
        "--phase-norm",
        action="store_true",
        default=True,
        help="Enable phase-aware robust normalization",
    )
    parser.add_argument(
        "--no-phase-norm",
        dest="phase_norm",
        action="store_false",
        help="Disable phase-aware robust normalization",
    )
    parser.add_argument(
        "--enable-stage0-liver-adapt",
        action="store_true",
        default=False,
        help="Enable Stage-0 liver-awareness warm-up before multitask fine-tuning (default: disabled)",
    )
    parser.add_argument(
        "--disable-stage0-liver-adapt",
        dest="enable_stage0_liver_adapt",
        action="store_false",
        help="Disable Stage-0 liver-awareness warm-up (default behavior)",
    )
    parser.add_argument("--stage0-epochs", type=int, default=5,
                        help="Epochs for Stage-0 liver-awareness warm-up")
    parser.add_argument("--stage0-lr", type=float, default=5e-5,
                        help="Learning rate for Stage-0 liver-awareness warm-up")
    parser.add_argument("--stage0-batch-size", type=int, default=2,
                        help="Batch size for Stage-0 liver-awareness warm-up")
    parser.add_argument("--stage0-wd", type=float, default=1e-4,
                        help="Weight decay for Stage-0 liver-awareness warm-up")
    parser.add_argument(
        "--stage0-unfreeze-last-n",
        type=int,
        default=1,
        help=(
            "Number of last spatial/temporal backbone blocks to unfreeze in Stage-0. "
            "0 means only visual projection; -1 means all visual backbone blocks."
        ),
    )
    parser.add_argument(
        "--stage0-unfreeze-scope",
        type=str,
        default=None,
        choices=["visual_projection", "visual_projection_plus_last", "all_visual"],
        help="(Legacy) Unfreeze scope alias. Prefer --stage0-unfreeze-last-n.",
    )
    parser.add_argument(
        "--stage0-negative-root",
        type=str,
        default=str(repo_root / "datasets/dataset/valid"),
        help="Optional non-liver CT root for Stage-0 negatives; fallback uses pseudo negatives when absent",
    )
    parser.add_argument("--stage0-max-negatives", type=int, default=64,
                        help="Max number of external negative scans for Stage-0")
    parser.add_argument(
        "--stage0-use-pseudo-negatives",
        action="store_true",
        default=True,
        help="Use non-liver pseudo-crops from liver scans when external negatives are missing",
    )
    parser.add_argument(
        "--no-stage0-use-pseudo-negatives",
        dest="stage0_use_pseudo_negatives",
        action="store_false",
        help="Disable pseudo negatives for Stage-0",
    )
    parser.add_argument(
        "--stage0-prompt",
        type=str,
        default="This is a preoperative abdominal contrast-enhanced CT volume for organ localization.",
        help="Text prompt used in Stage-0 liver-awareness warm-up",
    )
    parser.add_argument("--save-model", default=None,
                        help="Path to save regression model")
    parser.add_argument("--load-model", default=None,
                        help="Path to load pretrained regression model")
    parser.add_argument(
        "--split-file",
        default=None,
        help="Path to fixed split manifest JSON. Required for regression test stage.",
    )
    parser.add_argument("--stage", choices=["train", "test", "both"], default="test",
                        help="Stage flag (regression in this script is test-only; use ct_lipro_train.py for training)")

    # Model parameters
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--patch-size", type=int, default=20)

    return parser.parse_args()


def get_default_paths(dataset_type):
    """Get default paths based on dataset type."""
    base_path = str(Path(__file__).resolve().parents[1])

    if dataset_type == "chest":
        return {
            "data_folder": os.path.join(base_path, "datasets/dataset/valid"),
            "reports_file": os.path.join(base_path, "datasets/dataset/radiology_text_reports/validation_reports.csv"),
            "meta_file": os.path.join(base_path, "datasets/dataset/metadata/validation_metadata.csv"),
            "labels_file": os.path.join(base_path, "datasets/dataset/multi_abnormality_labels/valid_predicted_labels.csv"),
            "results_folder": os.path.join(base_path, "inference_zeroshot/"),
        }
    elif dataset_type == "hcc":
        return {
            "data_folder": os.path.join(base_path, "datasets/dataset/HCC_prepared/valid"),
            "reports_file": os.path.join(base_path, "datasets/dataset/HCC_prepared/radiology_text_reports/validation_reports.csv"),
            "meta_file": os.path.join(base_path, "datasets/dataset/HCC_prepared/metadata/validation_metadata.csv"),
            "labels_file": os.path.join(base_path, "datasets/dataset/HCC_prepared/multi_abnormality_labels/valid_predicted_labels.csv"),
            "results_folder": os.path.join(base_path, "inference_hcc/"),
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def run_classification(args):
    """Run zero-shot classification."""
    default_paths = get_default_paths(args.dataset_type)

    # Override with user-specified paths
    data_folder = args.data_folder or default_paths["data_folder"]
    reports_file = args.reports_file or default_paths["reports_file"]
    meta_file = args.meta_file or default_paths["meta_file"]
    labels_file = args.labels_file or default_paths["labels_file"]
    results_folder = args.results_folder or default_paths["results_folder"]

    print(f"Task: classification")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Data folder: {data_folder}")
    print(f"Reports file: {reports_file}")
    print(f"Meta file: {meta_file}")
    print(f"Labels file: {labels_file}")
    print(f"Results folder: {results_folder}")

    # Initialize tokenizer and text encoder
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialize image encoder
    image_encoder = CTViT(
        dim=args.dim,
        codebook_size=args.codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        temporal_patch_size=args.temporal_patch_size,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )

    # Initialize CLIP model
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=294912,
        dim_text=768,
        dim_latent=args.dim,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False
    )

    # Load checkpoint
    pt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = clip.load_state_dict(pt, strict=False)
    print(f"Loaded checkpoint with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # Create inference object
    inference = CTClipInference(
        clip,
        data_folder=data_folder,
        reports_file=reports_file,
        meta_file=meta_file,
        labels=labels_file,
        results_folder=results_folder,
    )

    # Run inference
    inference.infer()
    print("Zero-shot classification complete!")


def main():
    args = parse_args()

    if args.task == "regression":
        if args.stage != "test":
            raise SystemExit(
                "run_zero_shot.py in regression mode is test-only to avoid accidental weight overwrite. "
                "Use `python scripts/ct_lipro_train.py --task regression ...` for training."
            )
        run_regression(args)
    else:
        run_classification(args)


if __name__ == "__main__":
    main()
