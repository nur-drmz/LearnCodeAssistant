from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "ytu-ce-cosmos/turkish-gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_answer(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Örnek kullanım
soru = "Python'da döngüler nedir?"
cevap = generate_answer(soru)
print(cevap)
