import os
import torch
from datasets import load_dataset
from vlmeval import evaluate
from vlmeval.models import register_model
from solo_adapter import SOLOEvalAdapter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 所有可调整的配置都在这里 ============
EVALUATION_CONFIG = {
    # 模型相关配置
    "model": {
        "type": "solo",
        "name": "SOLO-7B",
        "path": "YangyiYY/SOLO-7B",
        "max_new_tokens": 64,        # 这里可以调整生成的最大token数
        "device": "cuda",
    },
    
    # 数据集配置
    "dataset": {
        "name": "MMStar",
        "type": "huggingface",
        "batch_size": 32,           # batch size大小
        "num_workers": 16,          # 数据加载的工作进程数
        "gen_mode": "mm",           # 'mm' 或 'to'
        "metrics": [
            "accuracy",
            "rouge",
            "bleu",
            "cider"
        ],
    },
    
    # 输出配置
    "output": {
        "save_predictions": True,
        "save_metrics": True
    },
    
    # 分布式训练配置
    "distributed": {
        "enabled": True,
        "world_size": torch.cuda.device_count(),
        "backend": "nccl"
    }
}

def setup_directories():
    """设置必要的目录"""
    base_dir = "/p/project1/westai0019"
    dirs = {
        'cache': os.path.join(base_dir, "models"),
        'dataset': os.path.join(base_dir, "datasets"),
        'output': os.path.join(base_dir, "evaluation_results", 
                              datetime.now().strftime("%Y%m%d_%H%M%S"))
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def main():
    # 设置目录
    dirs = setup_directories()
    
    # 设置GPU
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPUs")
    
    try:
        # 加载数据集
        dataset = load_dataset(
            "Lin-Chen/MMStar", 
            "val",
            cache_dir=dirs['dataset']
        )
        logger.info(f"Dataset loaded with {len(dataset['val'])} samples")
        
        # 注册模型
        register_model("solo", SOLOEvalAdapter)
        
        # 构建完整的评估配置
        config = {
            "model": EVALUATION_CONFIG["model"],
            "datasets": [{
                "name": EVALUATION_CONFIG["dataset"]["name"],
                "type": EVALUATION_CONFIG["dataset"]["type"],
                "dataset": dataset['val'],
                "metrics": EVALUATION_CONFIG["dataset"]["metrics"],
                "batch_size": EVALUATION_CONFIG["dataset"]["batch_size"],
                "num_workers": EVALUATION_CONFIG["dataset"]["num_workers"],
                "gen_mode": EVALUATION_CONFIG["dataset"]["gen_mode"],
            }],
            "output": {
                "path": dirs['output'],
                **EVALUATION_CONFIG["output"]
            },
            "distributed": EVALUATION_CONFIG["distributed"]
        }
        
        # 运行评估
        logger.info(f"Starting evaluation in {config['datasets'][0]['gen_mode']} mode...")
        logger.info(f"Using batch_size: {config['datasets'][0]['batch_size']}")
        logger.info(f"Using num_workers: {config['datasets'][0]['num_workers']}")
        logger.info(f"Using max_new_tokens: {config['model']['max_new_tokens']}")
        
        results = evaluate(config)
        
        # 保存结果
        results_file = os.path.join(
            dirs['output'], 
            f"evaluation_results_{config['datasets'][0]['gen_mode']}.txt"
        )
        
        with open(results_file, 'w') as f:
            f.write(f"Evaluation Results ({config['datasets'][0]['gen_mode']} mode):\n")
            f.write("=" * 50 + "\n")
            for dataset_name, metrics in results.items():
                f.write(f"\nDataset: {dataset_name}\n")
                for metric_name, score in metrics.items():
                    result_line = f"{metric_name}: {score:.4f}"
                    f.write(result_line + "\n")
                    logger.info(result_line)
        
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()