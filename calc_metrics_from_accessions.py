import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PATHOLOGIES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


def parse_accession_key(name: str):
    # name like "ID01.02555191_1" -> ("01", "02555191")
    prefix = name.split("_")[0]
    if "." in prefix:
        id_part, name_part = prefix.split(".", 1)
        id_part = id_part.replace("ID", "")
        return id_part, name_part
    return None, None


def load_accessions(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compute AUROC from accessions + HCC excel labels.")
    parser.add_argument(
        "--accessions",
        default="/root/autodl-tmp/CT-CLIP/inference_hcc/accessions.txt",
    )
    parser.add_argument(
        "--preds",
        default="/root/autodl-tmp/CT-CLIP/inference_hcc/predicted_weights.npz",
    )
    parser.add_argument(
        "--excel",
        default="/root/autodl-tmp/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx",
    )
    parser.add_argument(
        "--label-cols",
        default="坏死比例分组",
        help="Comma-separated HCC Excel label columns (binary).",
    )
    parser.add_argument(
        "--pred-index",
        type=int,
        default=-1,
        help="If set (0-17), only compute for this pathology index.",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/CT-CLIP/inference_hcc/metrics_from_accessions.xlsx",
    )
    args = parser.parse_args()

    accessions = load_accessions(args.accessions)
    preds = np.load(args.preds)["data"]
    if preds.shape[0] != len(accessions):
        raise ValueError(f"Pred rows {preds.shape[0]} != accessions {len(accessions)}")

    df = pd.read_excel(args.excel)
    label_cols = [c.strip() for c in args.label_cols.split(",") if c.strip()]

    results = []

    # Build lookup by NAME and ID
    by_name = {}
    by_id = {}
    if "NAME" in df.columns:
        for _, row in df.iterrows():
            try:
                name_key = int(row["NAME"])
            except Exception:
                continue
            by_name.setdefault(name_key, row)
    if "ID" in df.columns:
        for _, row in df.iterrows():
            try:
                id_key = int(row["ID"])
            except Exception:
                continue
            by_id.setdefault(id_key, row)

    for label_col in label_cols:
        if label_col not in df.columns:
            raise ValueError(f"Label column not found: {label_col}")

        y_true = []
        for acc in accessions:
            id_part, name_part = parse_accession_key(acc)
            row = None
            if name_part:
                try:
                    name_key = int(name_part.lstrip("0") or "0")
                    row = by_name.get(name_key)
                except Exception:
                    row = None
            if row is None and id_part:
                try:
                    id_key = int(id_part.lstrip("0") or "0")
                    row = by_id.get(id_key)
                except Exception:
                    row = None

            if row is None:
                y_true.append(None)
                continue

            val = row[label_col]
            if pd.isna(val):
                y_true.append(None)
            else:
                y_true.append(int(val))

        y_true = np.array(y_true, dtype=object)

        indices = range(len(PATHOLOGIES))
        if args.pred_index >= 0:
            indices = [args.pred_index]

        for i in indices:
            y_pred = preds[:, i]
            mask = np.array([v is not None for v in y_true])
            yt = y_true[mask].astype(int)
            yp = y_pred[mask]
            pos = int((yt == 1).sum())
            neg = int((yt == 0).sum())
            if pos == 0 or neg == 0:
                auc = np.nan
            else:
                auc = float(roc_auc_score(yt, yp))
            results.append(
                {
                    "label_col": label_col,
                    "pathology_index": i,
                    "pathology": PATHOLOGIES[i],
                    "auroc": auc,
                    "n": int(len(yt)),
                    "pos": pos,
                    "neg": neg,
                }
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)

    try:
        out_df.to_excel(out_path, index=False)
        print(f"wrote {out_path}")
    except Exception:
        csv_path = out_path.with_suffix(".csv")
        out_df.to_csv(csv_path, index=False)
        print(f"excel_write_failed, wrote {csv_path}")


if __name__ == "__main__":
    main()
