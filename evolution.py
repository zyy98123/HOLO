
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# 自定义图像处理函数

def load_image_to_base64(image_path):
    """
    将本地图像加载为base64编码的字符串。
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_str

def load_base64_to_PILImage(base64_str):
    """
    将base64字符串转换为PIL图像。
    """
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def convert_image_base64_to_patches(base64_str, patch_size=32):
    """
    将base64字符串代表的图像转换为图像块（patches）。
    """
    image = load_base64_to_PILImage(base64_str)
    image = image.convert('RGB')
    image = image.resize((patch_size * (image.width // patch_size), patch_size * (image.height // patch_size)))
    n_rows = image.height // patch_size
    n_cols = image.width // patch_size
    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            patch = image.crop((col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size))
            patches.append(np.array(patch).flatten())
    return torch.tensor(patches).view(n_rows, n_cols, -1)

def visualize_patches(patches, figsize=(8, 8)):
    """
    可视化图像块。
    """
    n_rows, n_cols, _ = patches.shape
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].imshow(patches[i, j].view(32, 32, 3).numpy().astype(np.uint8))
            axes[i, j].axis('off')
    plt.show()

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
            tokens.extend(["[INST]", i, "[/INST]"])
            attention_masks.extend([1] * (len(tokens) - len(attention_masks)))
    
    return torch.tensor(tokens).to(device), torch.tensor(attention_masks).to(device), torch.cat(vision_patches, dim=0).to(device), torch.tensor(vision_patch_indices).to(device)

# 加载和处理本地图像
def load_image_as_patches(image_path):
    img_base64 = load_image_to_base64(image_path)
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
    IMAGE_PATH = "./testIMG.png"

    model, tokenizer = load_model_and_tokenizer(DEVICE)
    img_patches = load_image_as_patches(IMAGE_PATH)

    inputs = [
        img_patches,
        "This is a"
    ]
    run_inference_and_print_outputs(model, tokenizer, inputs, DEVICE)
