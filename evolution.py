import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
from PIL import Image

# 导入图像处理工具
from image_utils import (
    load_base64_to_PILImage,
    convert_image_base64_to_patches,
    visualize_patches
)

# 设置模型路径并从Hugging Face加载模型
def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("YangyiYY/SOLO-7B")
    model = AutoModelForCausalLM.from_pretrained("YangyiYY/SOLO-7B", torch_dtype=torch.bfloat16)
    model = model.to(device)
    return model, tokenizer

# 准备输入数据
def prepare_inputs(inputs: list, device: str):
    NON_VISION_TOKEN = -1
    tokens = []
    attention_masks = []
    vision_patch_indices = []
    vision_patches = []
    
    for i in inputs:
        if isinstance(i, torch.Tensor):
            patches = i
            n_rows, n_cols = patches.shape[:2]
            n_patches = n_rows * n_cols
            patches = patches.view(n_patches, -1)
            
            img_tokens = ["<vision>"]
            cur_patch_indices = [NON_VISION_TOKEN]
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx != 0 and col_idx == 0:
                        img_tokens.append(f"<vrow_sep>")
                        cur_patch_indices.append(NON_VISION_TOKEN)
                    img_tokens.append(f"<vpatch>")

            tokens.extend(img_tokens)
            vision_patch_indices.extend(cur_patch_indices)
            vision_patches.append(patches)
        else:
            tokens.extend([B_INST, i, E_INST])
            attention_masks.extend([1] * (len(tokens) - len(attention_masks)))
    
    return torch.tensor(tokens).to(device), torch.tensor(attention_masks).to(device), torch.cat(vision_patches, dim=0).to(device), torch.tensor(vision_patch_indices).to(device)

# 加载和处理本地图像
def load_image_as_patches(image_path):
    img = Image.open(image_path)
    img_base64 = load_base64_to_PILImage(img)
    img_patches = convert_image_base64_to_patches(img_base64)
    return img_patches

# 运行推理函数
def run_inference_and_print_outputs(model, tokenizer, inputs, device, do_sample=False, top_p=0.95, max_new_tokens=30):
    tokens, attention_masks, vision_patches, vision_patch_indices = prepare_inputs(inputs, device=device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokens.unsqueeze(0),
            attention_mask=attention_masks.unsqueeze(0),
            vision_patches=vision_patches,
            vision_patch_indices=vision_patch_indices.unsqueeze(0),
            generation_config=GenerationConfig(
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                suppress_tokens=[i for i in range(32000, len(tokenizer))],
            ),
        )
    visualize_outputs(inputs, tokens, outputs)

# 主程序
if __name__ == "__main__":
    DEVICE = "cuda:0"
    IMAGE_PATH = "path/to/your/image.jpg"

    model, tokenizer = load_model_and_tokenizer(DEVICE)
    img_patches = load_image_as_patches(IMAGE_PATH)

    inputs = [
        img_patches,
        "This is a"
    ]
    run_inference_and_print_outputs(model, tokenizer, inputs, DEVICE)