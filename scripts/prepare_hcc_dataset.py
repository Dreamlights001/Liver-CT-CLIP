"""
数据集准备脚本

将自定义 CT 数据集转换为 CT-CLIP 零样本分类所需的格式。

输入格式:
    src_root/
    ├── patient_001/
    │   ├── scan1.nii.gz
    │   └── scan2.nii.gz
    └── patient_002/
        └── scan1.nii.gz

输出格式:
    out_root/
    ├── valid/
    ├── radiology_text_reports/
    │   └── validation_reports.csv
    ├── metadata/
    │   └── validation_metadata.csv
    └── multi_abnormality_labels/
        └── valid_predicted_labels.csv

用法:
    python scripts/prepare_hcc_dataset.py \
        --src-root /path/to/your_dataset \
        --out-root /path/to/prepared_dataset
"""

import argparse
import csv
import glob
import math
import os

import nibabel as nib


def load_label_columns(label_template_csv: str):
    """从模板 CSV 加载标签列名。"""
    with open(label_template_csv, "r", newline="", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if len(header) < 2 or header[0] != "VolumeName":
        raise ValueError(f"Unexpected label header in {label_template_csv}")
    return header[1:]


def safe_float(x, default):
    """安全转换为浮点数。"""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def build(
    src_root: str,
    out_root: str,
    label_template_csv: str = None,
    label_cols: list = None,
):
    """
    构建数据集。

    Args:
        src_root: 源数据根目录，包含患者文件夹
        out_root: 输出根目录
        label_template_csv: 标签模板 CSV 文件（用于获取列名）
        label_cols: 标签列名列表（如果提供，则忽略 label_template_csv）
    """
    # 获取标签列
    if label_cols is None:
        if label_template_csv is None:
            # 使用默认的胸部 CT 标签
            label_cols = [
                'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
                'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
                'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule',
                'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
                'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation',
                'Bronchiectasis', 'Interlobular septal thickening'
            ]
        else:
            label_cols = load_label_columns(label_template_csv)

    # 创建输出目录
    data_root = os.path.join(out_root, "valid")
    reports_root = os.path.join(out_root, "radiology_text_reports")
    meta_root = os.path.join(out_root, "metadata")
    labels_root = os.path.join(out_root, "multi_abnormality_labels")

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(reports_root, exist_ok=True)
    os.makedirs(meta_root, exist_ok=True)
    os.makedirs(labels_root, exist_ok=True)

    meta_rows = []
    report_rows = []
    label_rows = []

    # 遍历患者文件夹
    patient_dirs = sorted([p for p in glob.glob(os.path.join(src_root, "*")) if os.path.isdir(p)])
    print(f"Found {len(patient_dirs)} patient directories")

    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        nii_files = sorted(glob.glob(os.path.join(patient_dir, "*.nii.gz")))
        if not nii_files:
            print(f"Warning: No .nii.gz files found in {patient_dir}")
            continue

        # 创建扫描目录
        scan_dir = os.path.join(data_root, patient_id, f"{patient_id}_a")
        os.makedirs(scan_dir, exist_ok=True)

        for nii_path in nii_files:
            base = os.path.basename(nii_path)
            # 生成新的文件名
            new_name = f"{patient_id}_{base}"
            dest_path = os.path.join(scan_dir, new_name)

            # 创建符号链接（如果不存在）
            if not os.path.exists(dest_path):
                os.symlink(nii_path, dest_path)

            # 读取 NIfTI 文件元数据
            try:
                img = nib.load(nii_path)
                zooms = img.header.get_zooms()
                xy_spacing = safe_float(zooms[0], 1.0)
                z_spacing = safe_float(zooms[2], 1.0) if len(zooms) > 2 else 1.0

                slope = safe_float(img.header.get("scl_slope"), 1.0)
                intercept = safe_float(img.header.get("scl_inter"), 0.0)
            except Exception as e:
                print(f"Warning: Failed to read metadata from {nii_path}: {e}")
                xy_spacing = 1.0
                z_spacing = 1.0
                slope = 1.0
                intercept = 0.0

            # 添加元数据行
            meta_rows.append({
                "VolumeName": new_name,
                "XYSpacing": f"[{xy_spacing}, {xy_spacing}]",
                "RescaleIntercept": intercept,
                "RescaleSlope": slope,
                "ZSpacing": z_spacing,
            })

            # 添加报告行（空内容）
            report_rows.append({
                "VolumeName": new_name,
                "Findings_EN": "",
                "Impressions_EN": "",
            })

            # 添加标签行（全 0）
            label_row = {"VolumeName": new_name}
            for col in label_cols:
                label_row[col] = 0
            label_rows.append(label_row)

    # 写入 CSV 文件
    reports_path = os.path.join(reports_root, "validation_reports.csv")
    meta_path = os.path.join(meta_root, "validation_metadata.csv")
    labels_path = os.path.join(labels_root, "valid_predicted_labels.csv")

    with open(reports_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["VolumeName", "Findings_EN", "Impressions_EN"])
        writer.writeheader()
        writer.writerows(report_rows)

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["VolumeName", "XYSpacing", "RescaleIntercept", "RescaleSlope", "ZSpacing"]
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["VolumeName", *label_cols])
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nPrepared {len(meta_rows)} scans from {len(patient_dirs)} patients")
    print(f"\nOutput files:")
    print(f"  data_folder: {data_root}")
    print(f"  reports_file: {reports_path}")
    print(f"  meta_file: {meta_path}")
    print(f"  labels_file: {labels_path}")

    return {
        "data_folder": data_root,
        "reports_file": reports_path,
        "meta_file": meta_path,
        "labels": labels_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="准备自定义 CT 数据集以用于 CT-CLIP 零样本分类。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认标签列
  python scripts/prepare_hcc_dataset.py \\
      --src-root /path/to/your_dataset \\
      --out-root /path/to/prepared

  # 使用自定义标签模板
  python scripts/prepare_hcc_dataset.py \\
      --src-root /path/to/your_dataset \\
      --out-root /path/to/prepared \\
      --label-template /path/to/labels.csv

  # 为 HCC 数据集准备
  python scripts/prepare_hcc_dataset.py \\
      --src-root /home/wang/CT-CLIP/datasets/dataset/HCC \\
      --out-root /home/wang/CT-CLIP/datasets/dataset/HCC_prepared
        """
    )
    parser.add_argument(
        "--src-root",
        required=True,
        help="源数据根目录，包含患者文件夹（每个文件夹内有 .nii.gz 文件）"
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="输出根目录"
    )
    parser.add_argument(
        "--label-template",
        default="/home/wang/CT-CLIP/datasets/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
        help="标签模板 CSV 文件（用于获取列名）"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src_root):
        print(f"Error: Source directory does not exist: {args.src_root}")
        return 1

    paths = build(args.src_root, args.out_root, args.label_template)

    print("\n你可以使用以下命令运行零样本分类:")
    print(f"  python scripts/run_zero_shot.py --task classify \\")
    print(f"      --data-folder {paths['data_folder']} \\")
    print(f"      --reports-file {paths['reports_file']} \\")
    print(f"      --meta-file {paths['meta_file']} \\")
    print(f"      --labels-file {paths['labels']}")

    return 0


if __name__ == "__main__":
    exit(main())
