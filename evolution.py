from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
import requests

# Step 1: 加载模型和 Tokenizer
model_name = "YangyiYY/SOLO-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: 准备图像输入（例如从 URL 加载图像）
image_url = "https://datasets-server.huggingface.co/assets/Lin-Chen/MMStar/--/bc98d668301da7b14f648724866e57302778ab27/--/val/val/0/image/image.jpg?Expires=1729783176&Signature=k3UjHvKqS35GHXB~mazpbIgeEXjqPgNd5kz~g7ifzLp85-9lRoUB4fy3jn7cGlKUcpvrA2mh48s~Ze4I-bZhOMtowMqEXmS-lztSnvJ5kGOE1fcR1RYe3vrubC-nCkk5cJi-bOzCARjcYONDxMxt6zGlXQ02~1PHGmshO-ChzNPwze06s68l~sNGyO8TOCtjStp5RAIYWZ-o48gVgEQxcZeI~zgRgWQ1niWURJ3Px3HqwXn7ZYptxlp1gfrxFTl1-rpjsUUSC5SrWLikmCBN1RMqjuLTQBG8TZkWKOdHd7li96Ub8t1QAECWIyPtIhdhIeyFR1j-CkdrGMzgBv5VoA__&Key-Pair-Id=K3EI6M078Z3AC3"
image = Image.open(requests.get(image_url, stream=True).raw)

# 假设模型需要进行一些图像预处理（例如将图像编码为模型可接受的格式）
# 这里会依赖模型是否需要专门的图像预处理代码

# Step 3: 准备推理的文本输入（如果有）
text_input = "Which option describe the object relationship in the image correctly? Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book."

# Step 4: 组合输入并进行推理
# 假设模型接受图像和文本的联合输入
inputs = tokenizer(text_input, return_tensors="pt")

# 如果模型接受图像输入（需要看模型的要求），则输入图像
# 这里假设模型是视觉语言模型，需要组合图像和文本作为输入
outputs = model.generate(**inputs)

# Step 5: 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Output:", output_text)
