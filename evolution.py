import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 导入图像处理工具
from image_utils import (
    load_image_to_base64,
    convert_image_base64_to_patches
)

# 设置模型路径并从Hugging Face加载模型
def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("YangyiYY/SOLO-7B")
    model = AutoModelForCausalLM.from_pretrained("YangyiYY/SOLO-7B", torch_dtype=torch.bfloat16)
    model = model.to(device)
    return model, tokenizer

# 准备输入数据，包括文本和视觉输入
def prepare_inputs(inputs: list, device: str, tokenizer):
    B_INST, E_INST = tokenizer.convert_tokens_to_ids("<INST>"), tokenizer.convert_tokens_to_ids("</INST>")
    tokens = []
    attention_masks = []
    vision_patches = []

    for i in inputs:
        if isinstance(i, torch.Tensor):
            # 确保图像patch的形状符合模型的要求
            patches = i.view(-1, i.shape[-1]) if len(i.shape) == 3 else i  # 展平最后两维，确保兼容性
            vision_patches.append(patches)
            img_tokens = [tokenizer.convert_tokens_to_ids("<vision>")] * patches.shape[0]
            tokens.extend(img_tokens)
            attention_masks.extend([1] * len(img_tokens))
        else:
            # 文本输入处理
            text_tokens = tokenizer(i, return_tensors='pt').input_ids.squeeze(0).tolist()
            tokens.extend([B_INST] + text_tokens + [E_INST])
            attention_masks.extend([1] * len([B_INST] + text_tokens + [E_INST]))

    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long).to(device)

    if vision_patches:
        vision_patches_tensor = torch.cat(vision_patches, dim=0).to(device)
    else:
        vision_patches_tensor = None

    return tokens_tensor, attention_masks_tensor, vision_patches_tensor

# 加载和处理本地图像
def load_image_as_patches(image_path):
    img_base64 = load_image_to_base64(image_path)
    img_patches = convert_image_base64_to_patches(img_base64)
    return img_patches

# 运行推理函数
def run_inference_and_print_outputs(model, tokenizer, inputs, device, do_sample=False, top_p=0.95, max_new_tokens=30):
    tokens, attention_masks, vision_patches = prepare_inputs(inputs, device=device, tokenizer=tokenizer)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokens.unsqueeze(0),
            attention_mask=attention_masks.unsqueeze(0),
            generation_config=GenerationConfig(
                do_sample=do_sample,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                suppress_tokens=[i for i in range(32000, len(tokenizer))],
            ),
        )
    # 简单打印模型生成的输出
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 主程序
if __name__ == "__main__":
    DEVICE = "cuda:0"
    model, tokenizer = load_model_and_tokenizer(DEVICE)

    img_patches = load_image_as_patches("./testIMG.png")
    text_input = "This is a"
    inputs = [img_patches, text_input]

    run_inference_and_print_outputs(model, tokenizer, inputs, DEVICE)