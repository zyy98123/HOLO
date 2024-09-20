from datasets import load_dataset

# 加载数据集
ds = load_dataset("X2FD/LVIS-Instruct4V", cache_dir='/p/project1/westai0019/fineTuningDataset/detailedImageCaption')
print(ds)

# 检查数据集的分割
if "train" in ds:
    train_data = ds['train']
else:
    print("数据集中没有 'train' 分割")

# 访问 'validation' 分割
if "validation" in ds:
    validation_data = ds['validation']
    print(f"Validation 数据集共有 {len(validation_data)} 条记录")
else:
    print("数据集中没有 'validation' 分割")