import pandas as pd
import os

excel_path = "/root/autodl-tmp/CT-CLIP/datasets/dataset/HCC预实验.xlsx"

df = pd.read_excel(excel_path)

print("=== 数据基本信息 ===")
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print()

print("=== 所有列名 ===")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
print()

print("=== 前5行数据 ===")
print(df.head())
print()

print("=== 每列的数据类型 ===")
print(df.dtypes)
print()

print("=== 文本类列（可用于文本输入）===")
text_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        non_empty = df[col].dropna().astype(str).str.strip().replace('', pd.NA).dropna()
        if len(non_empty) > 0:
            text_cols.append(col)
            print(f"\n列 '{col}':")
            print(f"  非空数量: {len(non_empty)}/{len(df)}")
            print(f"  示例: {non_empty.iloc[0][:100]}...")

print("\n=== 数值类列 ===")
num_cols = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        num_cols.append(col)
        non_empty = df[col].dropna()
        print(f"{col}: 范围 [{non_empty.min()}, {non_empty.max()}], 均值 {non_empty.mean():.2f}")
