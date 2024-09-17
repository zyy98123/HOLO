# 需要首先安装 datasets 库
# pip install datasets

from datasets import load_dataset

# 加载数据集
dataset_name = "BAAI/CapsFusion-120M"

try:
    # 尝试加载数据集
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/WebSight", "v0.1", cache_dir='/p/project1/westai0019/cache')
    print("数据集加载成功！")
    print(f"数据集共有 {len(ds)} 条记录")

    # 打印数据集的前几条数据
    print(ds['train'][0:5])

except Exception as e:
    print(f"加载数据集时出现问题: {e}")
