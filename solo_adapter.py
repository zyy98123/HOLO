from transformers import LlamaTokenizer, SoloForCausalLM
import torch
from vlmevalkit.models.base import BaseModel
from typing import List, Dict, Any
import numpy as np
from image_utils import (
    load_image_to_base64,
    convert_image_base64_to_patches
)

class SOLOEvalAdapter(BaseModel):
    """SOLO model adapter for VLMEvalKit"""
    
    def __init__(
        self,
        model_name_or_path: str = "YangyiYY/SOLO-7B",
        device: str = "cuda",
        max_new_tokens: int = 32,
        cache_dir: str = "/p/project1/westai0019/models"
    ):
        super().__init__()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        
        print(f"Loading tokenizer and model from {model_name_or_path}...")
        print(f"Using cache directory: {cache_dir}")
        
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir
        )
        self.model = SoloForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        ).to(device)
        
        print("Model loaded successfully!")
        
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        
    def prepare_inputs(self, inputs: list, device: str):
        """Prepare inputs for SOLO model"""
        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        
        for i in inputs:
            if isinstance(i, torch.Tensor):
                # 处理图像patches
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
                        cur_patch_indices.append(len(vision_patches) + row_idx * n_cols + col_idx)
                
                img_tokens.append("</vision>")
                cur_patch_indices.append(NON_VISION_TOKEN)
                
                cur_tokens = torch.Tensor(self.tokenizer.convert_tokens_to_ids(img_tokens))
                cur_attention_mask = [1] * len(cur_tokens)
                
                tokens.extend(cur_tokens)
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend(cur_patch_indices)
                vision_patches.extend(patches.numpy().astype(np.float16))
                
            elif isinstance(i, str):
                # 处理文本输入
                i = self.tokenizer.bos_token + f"{self.B_INST} {i.strip()} {self.E_INST}"
                _tokenized = self.tokenizer(i, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                
                tokens.extend(cur_tokens)
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        
        # 转换为tensor并移动到指定设备
        tokens = torch.Tensor(tokens).long().to(device)
        attention_masks = torch.Tensor(attention_masks).long().to(device)
        vision_patch_indices = torch.Tensor(vision_patch_indices).long().to(device)
        
        if len(vision_patches) > 0:
            vision_patches = torch.Tensor(vision_patches).bfloat16().to(device)
        else:
            vision_patches = None
            
        return tokens, attention_masks, vision_patches, vision_patch_indices

    def generate_responses(
        self,
        images: List[str],
        prompts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[str]:
        """Generate responses for the given images and prompts"""
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_images = images[i:i + batch_size]
            
            batch_responses = []
            for img_path, prompt in zip(batch_images, batch_prompts):
                try:
                    # 处理图像
                    img_base64 = load_image_to_base64(img_path)
                    img_patches = convert_image_base64_to_patches(img_base64)
                    
                    # 准备输入
                    inputs = [img_patches, prompt]
                    tokens, attention_masks, vision_patches, vision_patch_indices = self.prepare_inputs(
                        inputs, 
                        self.device
                    )
                    
                    # 生成回答
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=tokens.unsqueeze(0),
                            attention_mask=attention_masks.unsqueeze(0),
                            vision_patches=vision_patches,
                            vision_patch_indices=vision_patch_indices.unsqueeze(0),
                            generation_config=GenerationConfig(
                                do_sample=True,
                                top_p=0.95,
                                max_new_tokens=self.max_new_tokens,
                                pad_token_id=self.tokenizer.eos_token_id,
                                suppress_tokens=[i for i in range(32000, len(self.tokenizer))],
                            ),
                        )
                    
                    # 解码输出
                    response = self.tokenizer.decode(
                        outputs[0, len(tokens):],
                        skip_special_tokens=True
                    ).strip()
                    
                    batch_responses.append(response)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    batch_responses.append("")  # 添加空字符串作为错误样本的响应
            
            responses.extend(batch_responses)
        
        return responses

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        return inputs["input_ids"].to(self.device)
        
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            "model_name": "SOLO-7B",
            "model_path": "YangyiYY/SOLO-7B",
            "max_new_tokens": self.max_new_tokens,
            "device": self.device,
            "cache_dir": self.cache_dir
        }

    def __repr__(self):
        """String representation of the adapter"""
        return f"SOLOEvalAdapter(model_path={self.model_path}, device={self.device})"