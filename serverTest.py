import os
from datasets import load_dataset

#首先创建fineTuningDataset文件夹
dataset_dir = "fineTuningDataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Directory '{dataset_dir}' created.")
else:
    print(f"Directory '{dataset_dir}' already exists.")