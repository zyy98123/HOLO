import os
from datasets import load_dataset

# 1. 创建 stage2dataset 文件夹
dataset_dir = "stage2dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Directory '{dataset_dir}' created.")
else:
    print(f"Directory '{dataset_dir}' already exists.")

# 2. 定义要下载的数据集
datasets = [
    ("BAAI/CapsFusion-120M", None),
    ("HuggingFaceM4/WebSight", "v0.2"),
    ("HuggingFaceM4/WebSight", "v0.1"),
    ("pixparse/cc3m-wds", None),
    ("SALT-NLP/LLaVAR", None),
    ("howard-hou/OCR-VQA", None),
    ("DKYoon/SlimPajama-6B", None),
]

# 3. 下载 Hugging Face 数据集
for dataset_name, version in datasets:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=dataset_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {dataset_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=dataset_dir)
        print(f"Dataset {dataset_name} downloaded to {dataset_dir}.")

# 4. 创建 Detailed Captions 文件夹
detailed_captions_dir = os.path.join(dataset_dir, "Detailed Captions")
if not os.path.exists(detailed_captions_dir):
    os.makedirs(detailed_captions_dir)
    print(f"Directory '{detailed_captions_dir}' created.")
else:
    print(f"Directory '{detailed_captions_dir}' already exists.")

# 5. 下载指定的 tsv 文件
urls = [
    "https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250",
    "https://storage.cloud.google.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv?_ga=2.141047602.-1896153081.1529438250",
    "https://storage.cloud.google.com/conceptual-captions-v1-1-labels/Image_Labels_Subset_Train_GCC-Labels-training.tsv?_ga=2.234395421.-20118413.1607637118"
]

for url in urls:
    wget_command = f"wget '{url}' -P '{detailed_captions_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {detailed_captions_dir}")
