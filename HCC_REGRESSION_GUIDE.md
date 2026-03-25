# HCC 坏死率回归完整流程（CT-CLIP）

本指南面向当前仓库环境，目标是用肝脏 CT 数据预测坏死率（0–1 回归）。

## 1. 数据要求与结构
假设你的 HCC 数据放在：
```
/home/wang/CT-CLIP/datasets/dataset/HCC/
```
每个患者一个目录，目录名形如 `IDxx.xxxxxxxx`，目录内有同一患者的 2 个体积：
```
ID01.02555191/
  1.nii.gz
  2.nii.gz
```
序列期别约定（本项目统一）：
- `1.nii.gz`：动脉期
- `2.nii.gz`：门静脉期

坏死率等临床信息在 Excel：
```
/home/wang/CT-CLIP/datasets/dataset/HCC/HCC预实验.xlsx
```

## 2. 脚本说明
新增脚本：
```
/home/dlts/CT-CLIP/scripts/run_zero_shot.py
/home/dlts/CT-CLIP/scripts/ct_lipro_train.py
```
功能：
- `run_train.py`：原始 CT-CLIP 预训练入口（胸部 CT + 报告对比学习），不用于当前 HCC 小样本回归实验
- `ct_lipro_train.py --task regression`：当前 HCC 回归训练入口
- `run_zero_shot.py --task regression --stage test`：当前 HCC 回归测试入口
- 目标：多任务预测（主任务 `坏死比例分组`，副任务 `坏死比例`）
- 图像输入：NIfTI 体积，内部做重采样与裁剪
- 文本输入：Excel 除目标列外所有字段拼接为 `col:value`
- 肝脏识别增强（不做分割）：右上腹先验裁剪 + 肝窗融合 + 分期归一化
- Stage-0 暖启动：先做肝脏识别适配，再进入坏死分组/比例训练
- 训练入口：`ct_lipro_train.py --task regression`
- 测试入口：`run_zero_shot.py --task regression --stage test`（仅测试，避免误覆盖权重）
- 支持两种策略对比：
  - `--scan-handling distinguish`：区分动脉/门静脉（phase-aware）
  - `--scan-handling ignore`：不区分动脉/门静脉（phase-agnostic）
- 训练/测试一致性：训练阶段写出 `split_manifest.json` 和 `checkpoint_manifest.json`，测试阶段强制校验配置与划分一致，不一致直接报错

输出目录默认：
```
/home/wang/CT-CLIP/inference_hcc_regression/
```
包含：
- `predictions.csv`：每个样本/患者的真实值与预测值
- `run_meta.csv`：本次运行参数记录
- `train_patients.csv`：训练集患者列表
- `test_patients.csv`：测试集患者列表
- `regressor.pt`：回归头权重（默认保存路径）
- `train_config.json`：训练参数与配置签名快照（训练阶段）
- `train_lr_log.csv`：逐 step 学习率日志（训练阶段）
- `train_epoch_log.csv`：逐 epoch 的 loss/metric 学习曲线日志（训练阶段）

## 3. 运行前说明
- 当前环境若无 GPU，会自动退回 CPU（运行较慢）
- 脚本默认不修改代理；若代理导致 HuggingFace 报错，可显式加 `--disable-proxy` 仅在当前进程临时清理 `ALL_PROXY/all_proxy`
- `坏死比例` 在样例数据中分布很窄（目前约 0.9–1.0），回归可分性有限，属于医学标签本身的限制

## 4. 运行命令（示例）

### A. 当前四组模板命令（含无CT基线）
```bash
conda activate test
cd /home/dlts/CT-CLIP

python scripts/ct_lipro_train.py --task regression --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template arterial_only
python scripts/ct_lipro_train.py --task regression --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template arterial_portal
python scripts/ct_lipro_train.py --task regression --train-n 4 --epochs 20 --lr 1e-3 --scan-handling distinguish --prompt-template all_features
python scripts/ct_lipro_train.py --task regression --train-n 4 --epochs 20 --lr 1e-3 --prompt-template tumor_markers_text_only
```

其中 `tumor_markers_text_only` 不读取 `1.nii.gz/2.nii.gz`，只使用肿瘤标志物文本列：
- `手术切除前AFP`
- `手术切除前 PIVKA`
- `诊断时AFP`
- `诊断时PIVKA-II`

四组模板当前语义：
- `arterial_only`：仅动脉期 CT + 最小非泄露背景字段（年龄/性别）
- `arterial_portal`：动脉+门静脉 CT + 最小非泄露背景字段（年龄/性别）
- `all_features`：动脉+门静脉 CT + 全部非目标临床字段（排除坏死比例、坏死比例分组）
- `tumor_markers_text_only`：不使用 CT，仅使用肿瘤标志物文本

提示词生成方式：采用“字段级完整医学叙述句”，将具体特征值直接写入句子（例如年龄/性别），而非尾部统一拼接特征字符串。

### A1. 无分割“肝脏认识”增强（默认关闭，按需开启）
```bash
python scripts/ct_lipro_train.py --task regression \
  --prompt-template arterial_portal \
  --liver-prior-crop right_upper_abdomen \
  --liver-window \
  --phase-norm \
  --enable-stage0-liver-adapt \
  --stage0-epochs 5 \
  --stage0-lr 5e-5
```

说明：
- 不使用分割模型，只做解剖先验裁剪与窗宽窗位增强
- Stage-0 先训练肝脏识别适配头（轻量解冻视觉分支），再进入主任务训练
- `tumor_markers_text_only` 模板会自动关闭 CT 与 Stage-0
- 默认训练流程不包含 Stage-0；需显式传 `--enable-stage0-liver-adapt` 才会执行

如需关闭增强并回到基线：
```bash
python scripts/ct_lipro_train.py --task regression \
  --no-liver-prior-crop \
  --no-liver-window \
  --no-phase-norm \
  --disable-stage0-liver-adapt
```

### A2. 训练模式切换（默认 LiPro）
```bash
# 默认 LiPro（冻结 CLIP）
python scripts/ct_lipro_train.py --task regression --train-mode lipro

# 切到 VocabFine（端到端微调）
python scripts/ct_lipro_train.py --task regression --train-mode vocabfine
```

兼容旧参数（不推荐）：
- `--official-finetune` 等价于 `--train-mode vocabfine`
- `--no-official-finetune` 等价于 `--train-mode lipro`

### B. 训练与测试分离（推荐）
训练：
```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro \
  --train-n 4 \
  --epochs 20 \
  --lr 1e-3 \
  --scan-handling distinguish \
  --prompt-template arterial_only \
  --out-dir /home/wang/CT-CLIP/inference_hcc_regression
```

测试（加载训练权重）：
```bash
python scripts/run_zero_shot.py --task regression \
  --train-mode lipro \
  --train-n 4 \
  --scan-handling distinguish \
  --prompt-template arterial_only \
  --stage test \
  --load-model /home/wang/CT-CLIP/inference_hcc_regression/regressor.pt \
  --split-file /home/wang/CT-CLIP/inference_hcc_regression/split_manifest.json \
  --out-dir /home/wang/CT-CLIP/inference_hcc_regression
```

### C. 区分与不区分两期对比
```bash
python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro \
  --train-n 4 \
  --epochs 20 \
  --lr 1e-3 \
  --scan-handling distinguish \
  --prompt-template arterial_portal

python scripts/ct_lipro_train.py --task regression \
  --train-mode lipro \
  --train-n 4 \
  --epochs 20 \
  --lr 1e-3 \
  --scan-handling ignore \
  --prompt-template arterial_portal
```

## 5. 关键参数说明
- `--target-col`：回归目标列，默认 `坏死比例`
- `--group-col`：分组标签列，默认 `坏死比例分组`
- `--train-mode`：`lipro`（默认）或 `vocabfine`
- `--scan-handling`：`distinguish` 或 `ignore`
- `--prompt-template`：`arterial_only` / `arterial_portal` / `all_features` / `tumor_markers_text_only`
- `--liver-prior-crop`：`right_upper_abdomen`（默认）或 `none`
- `--liver-window` / `--no-liver-window`：是否启用肝窗融合
- `--phase-norm` / `--no-phase-norm`：是否启用分期归一化
- `--enable-stage0-liver-adapt` / `--disable-stage0-liver-adapt`：Stage-0 肝脏识别暖启动开关
- `--stage0-epochs` / `--stage0-lr` / `--stage0-batch-size`：Stage-0 训练参数
- `--stage0-unfreeze-last-n`：Stage-0 解冻 backbone 末尾层数（`0`=仅视觉投影，`1`=默认末1层，`-1`=全视觉backbone）
- `--stage0-negative-root`：外部非肝脏负样本目录（若不存在则可用伪负样本）
- `--train-n`：训练样本数（其余为测试）
- `--train-ratio`：训练比例（当未提供 `--train-n` 时生效）
- `--epochs`：训练轮数
- `--lr`：学习率
- `--loss-weight-group`：分组损失权重（默认 0.8）
- `--loss-weight-ratio`：比例损失权重（默认 0.2）
- `--group-threshold`：分组阈值（默认 0.5）
- `--use-text/--no-text`：是否使用 Excel 文本特征（默认启用）
- `--save-model`：保存回归权重路径
- `--load-model`：加载回归权重路径
- `--split-file`：测试时必须提供的固定划分文件（推荐训练目录下 `split_manifest.json`）
- `--stage`：在 `run_zero_shot.py` 回归模式下固定使用 `test`（训练请用 `ct_lipro_train.py --task regression`）

## 6. 结果查看
运行完成后查看：
```
/home/wang/CT-CLIP/inference_hcc_regression/predictions.csv
```
列包括：`patient, ratio_target, ratio_pred, group_target, group_prob, group_pred`

其中：
- `ratio_pred` 为分组一致性后导出比例（若 `group_pred=1` 则为 1.0；否则不超过 0.99）
- `group_prob` / `group_pred` 为分组预测结果

## 7. 常见问题
1. **HuggingFace 下载失败（socks 代理）**
   - 默认保留代理；若代理不稳定可在命令中加入 `--disable-proxy` 临时禁用 `ALL_PROXY/all_proxy`（仅当前进程）。
2. **CPU 运行慢**
   - 先用 `--train-n 4 --epochs 2` 验证流程，再放大。
3. **坏死率区分度不足**
   - 当前 `坏死比例` 数值集中，回归指标会受限，可考虑换更有跨度的目标列。
