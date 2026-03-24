# 百度网盘数据下载与接入零样本推理（HCC影像）

## 1. 先把分享链接保存到自己的网盘
说明：PyBaiduPan 使用“网盘路径”进行操作，因此需要先把分享链接保存到你的网盘，再用 CLI 下载。（推断）

步骤：
1. 浏览器打开分享链接，输入提取码。
2. 点击“保存到网盘”，记下保存路径（示例：`/apps/ct-clip/HCC影像`）。

## 2. 用 PyBaiduPan 下载（pip）
```bash
conda activate test
python -m pip install pyBaiduPan

# 首次使用：启动登录服务
BdPan
# 浏览器打开 http://127.0.0.1:25000 完成登录

# 查看网盘路径
BdPan list /
BdPan list /apps/ct-clip

# 下载到本地
BdPan download "/apps/ct-clip/HCC影像" "/home/wang/CT-CLIP/datasets/custom/HCC影像"
```

## 3. 用你的数据替换推理输入
推理代码依赖三类 CSV（文件名可自定义，但路径要改对）：
- `reports_file`：包含 `VolumeName`、`Findings_EN`、`Impressions_EN`
- `labels`：包含 `VolumeName` + 18 个病灶列（若无标注，可先全 0，但 AUROC 无意义）
- `meta_file`：包含 `VolumeName`、`XYSpacing`、`ZSpacing`、`RescaleSlope`、`RescaleIntercept`

`VolumeName` 必须与 `.nii.gz` 文件名一致。`data_folder` 内的层级需与推理代码一致：`data_folder/病人/扫描/*.nii.gz`。

将 `scripts/run_zero_shot.py` 里的路径改成你的数据路径，例如：
```python
data_folder = "/home/wang/CT-CLIP/datasets/custom/HCC影像/valid"
reports_file= "/home/wang/CT-CLIP/datasets/custom/HCC影像/validation_reports.csv"
meta_file = "/home/wang/CT-CLIP/datasets/custom/HCC影像/validation_metadata.csv"
labels = "/home/wang/CT-CLIP/datasets/custom/HCC影像/valid_predicted_labels.csv"
results_folder = "/home/wang/CT-CLIP/inference_hcc/"
```

如果你的分享包里没有以上 CSV，告诉我文件结构，我帮你生成最小可用版本。
