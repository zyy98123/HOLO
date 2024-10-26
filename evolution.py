import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# 导入图像处理工具
from image_utils import (
    load_image_to_base64,
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
def prepare_inputs(inputs: list, device: str, tokenizer):
    NON_VISION_TOKEN = -1
    B_INST, E_INST = tokenizer.convert_tokens_to_ids("<INST>"), tokenizer.convert_tokens_to_ids("</INST>")
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

            img_tokens = [tokenizer.convert_tokens_to_ids("<vision>")]
            cur_patch_indices = [NON_VISION_TOKEN]
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    if row_idx != 0 and col_idx == 0:
                        img_tokens.append(tokenizer.convert_tokens_to_ids("<vrow_sep>"))
                        cur_patch_indices.append(NON_VISION_TOKEN)
                    img_tokens.append(tokenizer.convert_tokens_to_ids("<vpatch>"))

            tokens.extend(img_tokens)
            vision_patch_indices.extend(cur_patch_indices)
            vision_patches.append(patches)
        else:
            # 将文本token化，而不是直接加入tokens列表中
            text_tokens = tokenizer(i, return_tensors='pt').input_ids.squeeze(0).tolist()
            tokens.extend([B_INST] + text_tokens + [E_INST])
            attention_masks.extend([1] * (len([B_INST]) + len(text_tokens) + len([E_INST])))
    
    return (
        torch.tensor(tokens, dtype=torch.long).to(device),
        torch.tensor(attention_masks, dtype=torch.long).to(device),
        torch.cat(vision_patches, dim=0).to(device) if vision_patches else None,
        torch.tensor(vision_patch_indices, dtype=torch.long).to(device) if vision_patch_indices else None
    )

# 加载和处理本地图像
def load_image_as_patches(image_path):
    img_base64 = load_image_to_base64(image_path)
    img_patches = convert_image_base64_to_patches(img_base64)
    return img_patches

# 运行推理函数
def run_inference_and_print_outputs(model, tokenizer, inputs, device, do_sample=False, top_p=0.95, max_new_tokens=30):
    tokens, attention_masks, vision_patches, vision_patch_indices = prepare_inputs(inputs, device=device, tokenizer=tokenizer)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokens.unsqueeze(0),
            attention_mask=attention_masks.unsqueeze(0),
            vision_patches=vision_patches,
            vision_patch_indices=vision_patch_indices.unsqueeze(0) if vision_patch_indices is not None else None,
            generation_config=GenerationConfig(
                do_sample=do_sample,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                suppress_tokens=[i for i in range(32000, len(tokenizer))],
            ),
        )
    # 简单打印模型生成的输出
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 主程序
if __name__ == "__main__":
    DEVICE = "cuda:0"
    IMAGE_PATH = "./testIMG.png"

    model, tokenizer = load_model_and_tokenizer(DEVICE)
    img_patches = load_image_as_patches(IMAGE_PATH)

    inputs = [
        img_patches,
        "This is a"
    ]
    run_inference_and_print_outputs(model, tokenizer, inputs, DEVICE)