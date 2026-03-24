# CT-CLIP 多数据集使用指南

本指南介绍如何在 CT-CLIP 项目中使用不同类型的 CT 数据集。

## 目录

1. [支持的 Tasks](#支持的-tasks)
2. [数据集类型](#数据集类型)
3. [零样本分类](#零样本分类)
4. [回归任务 (HCC 坏死率预测)](#回归任务-hcc-坏死率预测)
5. [准备自定义数据集](#准备自定义数据集)
6. [参数详解](#参数详解)

---

## 支持的 Tasks

CT-CLIP 现在支持两种任务类型：

| Task | 描述 | 用途 |
|------|------|------|
| `classify` | 零样本分类 | 使用预训练的 CT-CLIP 进行病理分类 |
| `regression` | 回归预测 | 预测连续值（如坏死率） |

## 数据集类型

### 1. Chest (原始胸部 CT 数据集)

原始 CT-CLIP 训练使用的数据集格式：

```
datasets/dataset/
├── valid/
│   ├── valid_1/
│   │   └── valid_1_a/
│   │       ├── valid_1_a_1.nii.gz
│   │       └── valid_1_a_2.nii.gz
│   └── valid_2/
│       └── ...
├── radiology_text_reports/
│   └── validation_reports.csv
├── metadata/
│   └── validation_metadata.csv
└── multi_abnormality_labels/
    └── valid_predicted_labels.csv
```

### 2. HCC (肝脏 CT 数据集)

用于 HCC 坏死率预测的数据集格式：

```
datasets/dataset/HCC/
├── ID01.02555191/
│   ├── 1.nii.gz
│   └── 2.nii.gz
├── ID02.02555192/
│   ├── 1.nii.gz
│   └── 2.nii.gz
└── HCC预实验.xlsx  # 包含临床数据和目标值
```

---

## 零样本分类

### 基本用法

```bash
# 使用胸部 CT 数据集（默认）
python scripts/run_zero_shot.py --task classify --dataset-type chest

# 使用 HCC 准备好的数据集
python scripts/run_zero_shot.py --task classify --dataset-type hcc
```

### 自定义路径

```bash
python scripts/run_zero_shot.py --task classify \
    --data-folder /path/to/your/data \
    --reports-file /path/to/reports.csv \
    --meta-file /path/to/metadata.csv \
    --labels-file /path/to/labels.csv \
    --results-folder /path/to/output
```

### 输出文件

分类任务会在结果目录生成：

- `aurocs.xlsx` - 各病理的 AUROC 指标
- `labels_weights.npz` - 真实标签
- `predicted_weights.npz` - 预测概率
- `accessions.txt` - 样本名称列表

---

## 回归任务 (HCC 坏死率预测)

### 基本用法

```bash
# 训练（默认 4-shot, 20 epochs, lr=1e-3）
python scripts/ct_lipro_train.py --task regression \
    --hcc-root /home/wang/CT-CLIP/datasets/dataset/HCC \
    --excel /home/wang/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx \
    --target-col "坏死比例" \
    --train-n 4 \
    --epochs 20 \
    --lr 1e-3 \
    --scan-handling distinguish \
    --prompt-template arterial_only

# 测试（显式 test-only）
python scripts/run_zero_shot.py --task regression \
    --stage test \
    --train-n 4 \
    --scan-handling distinguish \
    --prompt-template arterial_only \
    --load-model /home/wang/CT-CLIP/inference_hcc_regression/regressor.pt \
    --split-file /home/wang/CT-CLIP/inference_hcc_regression/split_manifest.json
```

### 推荐四组模板命令（含无CT基线）

```bash
python scripts/ct_lipro_train.py --task regression --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template arterial_only
python scripts/ct_lipro_train.py --task regression --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template arterial_portal
python scripts/ct_lipro_train.py --task regression --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template all_features
python scripts/ct_lipro_train.py --task regression --train-mode lipro --train-n 4 --epochs 20 --lr 1e-3 --prompt-template tumor_markers_text_only
```

`tumor_markers_text_only` 为纯文本消融模板，不使用 `1.nii.gz` / `2.nii.gz`，仅使用 Excel 中肿瘤标志物相关列：
- `手术切除前AFP`
- `手术切除前 PIVKA`
- `诊断时AFP`
- `诊断时PIVKA-II`

四组模板当前语义：
- `arterial_only`：仅动脉期 CT + 最小非泄露背景字段（年龄/性别/BMI）
- `arterial_portal`：动脉+门脉 CT + 最小非泄露背景字段（年龄/性别/BMI）
- `all_features`：动脉+门脉 CT + 全部非目标临床字段（排除坏死比例、坏死比例分组）
- `tumor_markers_text_only`：不使用 CT，仅使用肿瘤标志物文本

提示词生成方式：采用“字段级完整医学叙述句”，把具体变量写入句子（如“患者年龄为xx岁”），而不是统一拼接 `feature_text`。

### 训练模式切换（LiPro / VocabFine）

```bash
# 默认：LiPro（冻结 CLIP，仅训练回归头）
python scripts/ct_lipro_train.py --task regression --train-mode lipro

# 切换：VocabFine（端到端微调 CLIP + 回归头）
python scripts/ct_lipro_train.py --task regression --train-mode vocabfine
```

兼容旧参数（不推荐）：
- `--official-finetune` 等价于 `--train-mode vocabfine`
- `--no-official-finetune` 等价于 `--train-mode lipro`

### 任务定义（当前）

当前 HCC 训练为多任务：
- 主任务：`坏死比例分组`（0/1）
- 副任务：`坏死比例`（0-1）
- 默认损失权重：`0.8 * 分组BCE + 0.2 * 比例MSE`
- 导出结果时优先分组：若预测分组=1，则 `ratio_pred=1.0`；否则 `ratio_pred<=0.99`

### 训练和测试分离（推荐）

```bash
# 第一步：仅训练（默认 4-shot）
python scripts/ct_lipro_train.py --task regression \
    --hcc-root /home/wang/CT-CLIP/datasets/dataset/HCC \
    --excel /home/wang/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx \
    --target-col "坏死比例" \
    --train-n 4 \
    --epochs 20 \
    --lr 1e-3 \
    --scan-handling distinguish \
    --prompt-template arterial_only \
    --out-dir /home/wang/CT-CLIP/inference_hcc_regression

# 第二步：仅测试（加载训练好的模型）
python scripts/run_zero_shot.py --task regression \
    --hcc-root /home/wang/CT-CLIP/datasets/dataset/HCC \
    --excel /home/wang/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx \
    --target-col "坏死比例" \
    --train-n 4 \
    --scan-handling distinguish \
    --prompt-template arterial_only \
    --stage test \
    --load-model /home/wang/CT-CLIP/inference_hcc_regression/regressor.pt \
    --split-file /home/wang/CT-CLIP/inference_hcc_regression/split_manifest.json \
    --out-dir /home/wang/CT-CLIP/inference_hcc_regression
```

兼容方式：如需旧行为，可继续使用 `run_zero_shot.py --task regression --stage test` 做测试。

**重要提示**：
- 默认使用 4-shot 训练（`--train-n 4`）
- 默认训练轮数为 20（`--epochs 20`），学习率为 `1e-3`
- 默认训练模式为 LiPro（`--train-mode lipro`）
- 测试时必须使用训练产出的 `--split-file`（`split_manifest.json`）和对应 `--load-model`，以确保数据划分与 checkpoint 严格一致
- 训练阶段会保存 `train_patients.csv` 和 `test_patients.csv` 用于复现数据划分

### 扫描处理策略

当每个患者有多个扫描（如 1.nii.gz 和 2.nii.gz）时，有两种处理方式：

```bash
# 方式 1: 区分两期（phase-aware，病人为单位）
--scan-handling distinguish

# 方式 2: 不区分两期（phase-agnostic，病人为单位）
--scan-handling ignore
```

### 训练/测试划分

```bash
# 按样本数划分
--train-n 4  # 4 个病人训练，其余测试

# 按比例划分
--train-ratio 0.7  # 70% 训练，30% 测试
```

### 输出文件

回归任务会在输出目录生成：

- `predictions.csv` - 每个样本的预测值和真实值
- `train_patients.csv` - 训练集患者列表
- `test_patients.csv` - 测试集患者列表
- `regressor.pt` - 回归模型权重
- `run_meta.csv` - 运行参数记录
- `train_config.json` - 训练参数/配置签名快照（训练阶段）
- `train_lr_log.csv` - 学习率随 step 变化日志（训练阶段）
- `train_epoch_log.csv` - 每个 epoch 的 loss/metric 日志（训练阶段）

---

## 准备自定义数据集

### 步骤 1: 组织数据

将你的 CT 数据组织成以下格式：

```
your_dataset/
├── patient_001/
│   ├── scan1.nii.gz
│   └── scan2.nii.gz  # 可选
├── patient_002/
│   └── scan1.nii.gz
└── ...
```

### 步骤 2: 准备 Excel 文件

创建一个 Excel 文件，包含以下信息：

| ID | NAME | 坏死比例 | 年龄 | 性别 | ... |
|----|------|----------|------|------|-----|
| 1  | 2555191 | 0.95 | 65 | M | ... |
| 2  | 2555192 | 0.80 | 58 | F | ... |

- `ID`: 与文件夹名 `IDxx.xxx` 中的 `xx` 对应
- `NAME`: 与文件夹名 `IDxx.xxx` 中的 `xxx` 对应
- 其他列：可作为文本特征输入模型

### 步骤 3: 运行

```bash
python scripts/ct_lipro_train.py --task regression \
    --hcc-root /path/to/your_dataset \
    --excel /path/to/your_data.xlsx \
    --target-col "你的目标列名" \
    --out-dir /path/to/output
```

### 为分类准备数据

如果需要进行零样本分类（而非回归），需要准备以下 CSV 文件：

#### reports.csv
```csv
VolumeName,Findings_EN,Impressions_EN
patient_001_scan1.nii.gz,Some findings...,Some impressions...
```

#### metadata.csv
```csv
VolumeName,XYSpacing,RescaleIntercept,RescaleSlope,ZSpacing
patient_001_scan1.nii.gz,"[0.75, 0.75]",0.0,1.0,1.5
```

#### labels.csv
```csv
VolumeName,Medical material,Cardiomegaly,...
patient_001_scan1.nii.gz,0,1,...
```

可以使用 `prepare_hcc_dataset.py` 脚本自动生成这些文件：

```bash
python scripts/prepare_hcc_dataset.py \
    --src-root /path/to/your_dataset \
    --out-root /path/to/prepared_dataset \
    --label-template /path/to/label_template.csv
```

---

## 参数详解

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--task` | classify | 任务类型：classify 或 regression |
| `--checkpoint` | ckpt/CT-CLIP_v2.pt | CT-CLIP 权重路径 |
| `--dim` | 512 | 模型维度 |
| `--image-size` | 480 | 图像大小 |
| `--patch-size` | 20 | Patch 大小 |
| `--temporal-patch-size` | 10 | 时间维度 Patch 大小 |

### 分类参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-type` | chest | 数据集类型：chest 或 hcc |
| `--data-folder` | 自动 | 数据目录 |
| `--reports-file` | 自动 | 报告 CSV |
| `--meta-file` | 自动 | 元数据 CSV |
| `--labels-file` | 自动 | 标签 CSV |
| `--results-folder` | 自动 | 结果输出目录 |

### 回归参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hcc-root` | datasets/dataset/HCC | HCC 数据根目录 |
| `--excel` | HCC预实验.xlsx | 临床数据 Excel |
| `--target-col` | 坏死比例 | 目标列名 |
| `--group-col` | 坏死比例分组 | 分组标签列（0/1） |
| `--out-dir` | inference_hcc_regression | 输出目录 |
| `--train-mode` | lipro | 训练模式：lipro（冻结 CLIP）或 vocabfine（端到端） |
| `--scan-handling` | distinguish | 扫描处理方式（区分动脉/门脉） |
| `--prompt-template` | arterial_only | 提示模板：`arterial_only` / `arterial_portal` / `all_features` / `tumor_markers_text_only` |
| `--use-text` | True | 使用文本特征 |
| `--train-ratio` | None | 训练比例（当未提供 `--train-n` 时生效） |
| `--train-n` | 4 | 训练样本数（默认 4-shot，优先于 train-ratio） |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 1 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--loss-weight-group` | 0.8 | 分组任务损失权重（BCE） |
| `--loss-weight-ratio` | 0.2 | 比例任务损失权重（MSE） |
| `--group-threshold` | 0.5 | 分组概率阈值 |
| `--wd` | 0.1 | AdamW 权重衰减 |
| `--warmup-length` | 10 | cosine warmup 步数 |
| `--liver-prior-crop` | right_upper_abdomen | 非分割肝脏先验裁剪（`right_upper_abdomen` 或 `none`） |
| `--liver-window` | True | 肝窗融合预处理（可用 `--no-liver-window` 关闭） |
| `--phase-norm` | True | 分期鲁棒归一化（可用 `--no-phase-norm` 关闭） |
| `--enable-stage0-liver-adapt` | False | Stage-0 肝脏识别暖启动（默认关闭，需显式开启） |
| `--stage0-epochs` | 5 | Stage-0 训练轮数 |
| `--stage0-lr` | 5e-5 | Stage-0 学习率 |
| `--stage0-batch-size` | 2 | Stage-0 批大小 |
| `--stage0-unfreeze-last-n` | 1 | Stage-0 解冻 backbone 末尾层数（0=仅投影，-1=全视觉backbone） |
| `--stage0-negative-root` | datasets/dataset/valid | Stage-0 外部负样本目录（不存在时可退回伪负样本） |
| `--seed` | 42 | 随机种子 |
| `--official-finetune` | False | 兼容参数：等价于 `--train-mode vocabfine` |
| `--stage` | test | `run_zero_shot.py` 回归模式仅用于测试（推荐固定 `--stage test`） |
| `--save-model` | 自动 | 保存模型路径 |
| `--load-model` | None | 加载模型路径 |
| `--split-file` | None | 测试时固定划分文件（必填，避免训练/测试划分漂移） |
| `--target-depth` | 240 | 目标深度 |

---

## 常见问题

### 1. HuggingFace 下载失败

脚本默认不修改代理。若代理导致下载失败，可在命令中加 `--disable-proxy`，或手动设置：

```bash
unset ALL_PROXY
unset all_proxy
```

### 2. GPU 内存不足

可以减小 `--target-depth` 或使用 CPU：

```bash
--target-depth 120  # 减小深度
```

### 3. 数据集路径问题

确保路径正确，可以使用绝对路径：

```bash
python scripts/ct_lipro_train.py --task regression \
    --hcc-root "/absolute/path/to/HCC" \
    --excel "/absolute/path/to/HCC预实验.xlsx"
```

### 4. Excel 编码问题

确保 Excel 文件使用 UTF-8 编码，列名不要包含特殊字符。

---

## 示例工作流程

### 完整的 HCC 坏死率预测流程

```bash
# 1. 激活环境
conda activate test

# 2. 训练（默认不含 Stage-0；如需肝脏认识请显式加参数）
python scripts/ct_lipro_train.py --task regression \
    --hcc-root /home/wang/CT-CLIP/datasets/dataset/HCC \
    --excel /home/wang/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx \
    --target-col "坏死比例" \
    --scan-handling distinguish \
    --train-n 4 \
    --epochs 20 \
    --lr 1e-3 \
    --prompt-template arterial_only \
    --out-dir /home/wang/CT-CLIP/inference_hcc_regression

# 3. 测试（避免覆盖权重）
python scripts/run_zero_shot.py --task regression \
    --stage test \
    --train-n 4 \
    --scan-handling distinguish \
    --prompt-template arterial_only \
    --load-model /home/wang/CT-CLIP/inference_hcc_regression/regressor.pt \
    --split-file /home/wang/CT-CLIP/inference_hcc_regression/split_manifest.json \
    --out-dir /home/wang/CT-CLIP/inference_hcc_regression

# 4. 查看结果
cat /home/wang/CT-CLIP/inference_hcc_regression/predictions.csv
```

### 影像与无影像消融对比

```bash
# 影像 + 文本（双期）
python scripts/ct_lipro_train.py --task regression \
    --scan-handling distinguish \
    --prompt-template arterial_portal

# 仅肿瘤标志物文本（无CT）
python scripts/ct_lipro_train.py --task regression \
    --prompt-template tumor_markers_text_only
```

### 对比不同扫描处理策略

```bash
# 策略 1: 区分两期
python scripts/ct_lipro_train.py --task regression \
    --scan-handling distinguish \
    --out-dir results_distinguish

# 策略 2: 不区分两期
python scripts/ct_lipro_train.py --task regression \
    --scan-handling ignore \
    --out-dir results_ignore
```

---

## 文件结构

优化后的项目结构：

```
CT-CLIP/
├── scripts/
│   ├── run_zero_shot.py      # 统一推理入口（分类 + 回归测试）
│   ├── zero_shot.py           # 核心模块（CTClipInference + 回归工具）
│   ├── prepare_hcc_dataset.py # 数据集准备脚本
│   ├── hcc_necrosis_regression.py  # [保留] 独立回归脚本
│   └── run_zero_shot_hcc.py   # [已弃用] 使用 run_zero_shot.py 替代
├── datasets/
│   └── dataset/
│       ├── valid/             # 原始胸部 CT 数据
│       └── HCC/               # HCC 肝脏数据
├── ckpt/
│   └── CT-CLIP_v2.pt          # 预训练权重
├── inference_zeroshot/         # 分类结果
├── inference_hcc_regression/   # 回归结果
├── DATASET_GUIDE.md           # 本文档
└── HCC_REGRESSION_GUIDE.md    # HCC 回归详细指南
```

---

## 更多信息

- [HCC_REGRESSION_GUIDE.md](HCC_REGRESSION_GUIDE.md) - HCC 坏死率回归详细指南
- [README.md](README.md) - CT-CLIP 项目总览
- [scripts/README.md](scripts/README.md) - 脚本详细说明
