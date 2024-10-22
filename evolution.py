from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和 tokenizer
model_name = "YangyiYY/SOLO-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 构建一个简单的输入
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成输出
output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0]))
