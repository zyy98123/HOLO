from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
import requests

model_name = "YangyiYY/SOLO-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

image_url = "https://datasets-server.huggingface.co/assets/Lin-Chen/MMStar/--/bc98d668301da7b14f648724866e57302778ab27/--/val/val/0/image/image.jpg?Expires=1729783176&Signature=k3UjHvKqS35GHXB~mazpbIgeEXjqPgNd5kz~g7ifzLp85-9lRoUB4fy3jn7cGlKUcpvrA2mh48s~Ze4I-bZhOMtowMqEXmS-lztSnvJ5kGOE1fcR1RYe3vrubC-nCkk5cJi-bOzCARjcYONDxMxt6zGlXQ02~1PHGmshO-ChzNPwze06s68l~sNGyO8TOCtjStp5RAIYWZ-o48gVgEQxcZeI~zgRgWQ1niWURJ3Px3HqwXn7ZYptxlp1gfrxFTl1-rpjsUUSC5SrWLikmCBN1RMqjuLTQBG8TZkWKOdHd7li96Ub8t1QAECWIyPtIhdhIeyFR1j-CkdrGMzgBv5VoA__&Key-Pair-Id=K3EI6M078Z3AC3"
image = Image.open(requests.get(image_url, stream=True).raw)



prompt = "The image shows a suitcase and a book. Which option describes the object relationship in the image correctly? Choose from the following options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suitcase is beneath the bed., D: The suitcase is beneath the book. Please select the correct answer."


inputs = tokenizer(prompt, return_tensors="pt")


outputs = model.generate(**inputs, max_new_tokens=1000)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Output:", output_text)
