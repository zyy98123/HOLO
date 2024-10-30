import os
import torch
from vlmeval import evaluate_model
from transformers import LlamaTokenizer, SoloForCausalLM

def main():
    # 初始化模型和tokenizer
    model_path = "YangyiYY/SOLO-7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = SoloForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 设置评估配置
    eval_config = {
        "dataset": "mmstar",  # 使用MMStar数据集
        "split": "val",       # 使用验证集
        "batch_size": 32,     # 根据您的4个A100配置
        "num_workers": 16,
        "max_new_tokens": 64,
    }
    
    # 运行评估
    print("Starting evaluation...")
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        **eval_config
    )
    
    # 保存结果
    output_dir = "/p/project1/westai0019/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write("=" * 50 + "\n")
        for metric_name, score in results.items():
            result_line = f"{metric_name}: {score:.4f}"
            f.write(result_line + "\n")
            print(result_line)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()