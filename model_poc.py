from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(

    "mistralai/Mistral-7B-v0.3"

)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", padding_side="left")

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, orange, purple, pink,'