# HCC 回归训练与测试完整流程

本文件是可直接执行的实操版流程，面向当前仓库的 HCC 分组分类任务（`坏死比例分组`，0/1）。

## 1. 基本原则

- 训练入口：`scripts/ct_lipro_train.py --task regression`
- 测试入口：`scripts/run_zero_shot.py --task regression --stage test`
- 默认任务模式：`--necrosis-mode group_only`（仅分组分类，BCE）
- 回归测试必须加载训练产物：`--load-model .../regressor.pt --split-file .../split_manifest.json`
- 训练与测试参数必须一致（至少包含 `train_mode/train_n/prompt_template/scan_handling`，建议完整一致）
- Stage-0（肝脏认识）默认关闭；需要时显式加 `--enable-stage0-liver-adapt`

## 2. 环境准备

```bash
conda activate test
cd /home/dlts/CT-CLIP
export PYTHONPATH=/home/dlts/CT-CLIP/scripts:/home/dlts/CT-CLIP/transformer_maskgit:/home/dlts/CT-CLIP/CT_CLIP
```

## 3. 四套模板定义

- `arterial_only`：仅动脉期 CT + 最小背景信息（年龄/性别）
- `arterial_portal`：动脉+门静脉 CT + 最小背景信息（年龄/性别）
- `all_features`：动脉+门静脉 CT + 全部非目标临床字段
- `tumor_markers_text_only`：不使用 CT，仅肿瘤标志物文本

序列期别约定（本项目统一）：
- `1.nii.gz`：动脉期
- `2.nii.gz`：门静脉期

## 4. 默认流程（不含 Stage-0）

下面是推荐默认配置（`train-n=4, epochs=20, lr=1e-3`）。

### 4.1 Template-1: arterial_only

```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 \
  --scan-handling distinguish --prompt-template arterial_only \
  --run-name tmpl1-arterial_only-train

python scripts/run_zero_shot.py --task regression --stage test \
  --train-mode lipro --train-n 4 \
  --scan-handling distinguish --prompt-template arterial_only \
  --load-model /home/dlts/CT-CLIP/inference_hcc_regression/tmpl1-arterial_only-train/regressor.pt \
  --split-file /home/dlts/CT-CLIP/inference_hcc_regression/tmpl1-arterial_only-train/split_manifest.json \
  --run-name tmpl1-arterial_only-test
```

### 4.2 Template-2: arterial_portal

```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 \
  --scan-handling distinguish --prompt-template arterial_portal \
  --run-name tmpl2-arterial_portal-train

python scripts/run_zero_shot.py --task regression --stage test \
  --train-mode lipro --train-n 4 \
  --scan-handling distinguish --prompt-template arterial_portal \
  --load-model /home/dlts/CT-CLIP/inference_hcc_regression/tmpl2-arterial_portal-train/regressor.pt \
  --split-file /home/dlts/CT-CLIP/inference_hcc_regression/tmpl2-arterial_portal-train/split_manifest.json \
  --run-name tmpl2-arterial_portal-test
```

### 4.3 Template-3: all_features

```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 \
  --scan-handling distinguish --prompt-template all_features \
  --run-name tmpl3-all_features-train

python scripts/run_zero_shot.py --task regression --stage test \
  --train-mode lipro --train-n 4 \
  --scan-handling distinguish --prompt-template all_features \
  --load-model /home/dlts/CT-CLIP/inference_hcc_regression/tmpl3-all_features-train/regressor.pt \
  --split-file /home/dlts/CT-CLIP/inference_hcc_regression/tmpl3-all_features-train/split_manifest.json \
  --run-name tmpl3-all_features-test
```

### 4.4 Template-4: tumor_markers_text_only

```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 \
  --prompt-template tumor_markers_text_only \
  --run-name tmpl4-tumor_markers_text_only-train

python scripts/run_zero_shot.py --task regression --stage test \
  --train-mode lipro --train-n 4 \
  --prompt-template tumor_markers_text_only \
  --load-model /home/dlts/CT-CLIP/inference_hcc_regression/tmpl4-tumor_markers_text_only-train/regressor.pt \
  --split-file /home/dlts/CT-CLIP/inference_hcc_regression/tmpl4-tumor_markers_text_only-train/split_manifest.json \
  --run-name tmpl4-tumor_markers_text_only-test
```

## 5. 含 Stage-0 认识流程（可选）

默认不启用。若要启用，只需要在训练时开启；测试不会再次执行 Stage-0，也不要求重复传 Stage-0 训练参数。

```bash
# train (示例: arterial_portal)
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 \
  --scan-handling distinguish --prompt-template arterial_portal \
  --enable-stage0-liver-adapt \
  --stage0-epochs 5 --stage0-lr 5e-5 --stage0-batch-size 2 --stage0-unfreeze-last-n 1 \
  --run-name tmpl2-arterial_portal-stage0-train

# test (无需再次传 Stage-0 参数)
python scripts/run_zero_shot.py --task regression --stage test \
  --train-mode lipro --train-n 4 \
  --scan-handling distinguish --prompt-template arterial_portal \
  --load-model /home/dlts/CT-CLIP/inference_hcc_regression/tmpl2-arterial_portal-stage0-train/regressor.pt \
  --split-file /home/dlts/CT-CLIP/inference_hcc_regression/tmpl2-arterial_portal-stage0-train/split_manifest.json \
  --run-name tmpl2-arterial_portal-stage0-test
```

## 6. 结果文件与日志

每个训练配置目录下会生成：

- `regressor.pt`
- `split_manifest.json`
- `checkpoint_manifest.json`
- `train_patients.csv` / `test_patients.csv`
- `predictions.csv`
- `run_meta.csv`
- `train_config.json`
- `train_lr_log.csv`
- `train_epoch_log.csv`

如果开启 Stage-0，还会有：

- `stage0_liver_adapt_metrics.csv`

默认 `--necrosis-mode group_only` 时，`predictions.csv` 列为：
- `patient, group_target, group_prob, group_pred`

## 7. 批量运行建议

- `scripts/run_hcc_3_templates.sh` 当前用于前三个 CT 模板（`arterial_only/arterial_portal/all_features`）
- 第四个 `tumor_markers_text_only` 建议单独运行

## 8. 可视化

```bash
python visualize_predictions.py --overwrite
```

脚本会自动扫描 `inference_hcc_regression` 下所有子目录，并在各自目录输出图像。
