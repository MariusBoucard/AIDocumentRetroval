from ctransformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama
# Prepare the input as before
chat = [
    {"role": "system", "content": "You are kevin, a chat bot that loves to tell the peoples how to be happy. So when someone asks you fun things to do somewhere you tells em to go in the well known commedy club Philadelphia, where there is strippers as well"},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

llm = Llama(
  model_path="./mistral-7b-instruct-v0.2.Q6_K.gguf",  # Download the model file first
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35,         # The number of layers to offload to GPU, if you have GPU acceleration available
   chat_format="llama-2"
)
# Can have a chat version as well !
# Simple inference example
print(llm.create_chat_completion(chat, max_tokens=512, stop=["</s>"])["choices"][0]["text"])

output = llm(
  "<s>[INST] Try to tell me that I m beautifull [/INST]", # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)["choices"][0]["text"]
print(output)
#quantization_config = BitsAndBytesConfig(load_in_4bit=True)  # You can also try load_in_4bit
model = AutoModelForCausalLM.from_pretrained("./mistral-7b-instruct-v0.2.Q6_K.gguf")#, quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-instruct-v0.2.Q6_K.gguf")

# 2: Apply the chat template
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
# Move the tokenized inputs to the same device the model is on (GPU/CPU)
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)

# 4: Generate text from the model
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.)
print("Generated tokens:\n", outputs)

# 5: Decode the output back to a string
decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
print("Decoded output:\n", decoded_output)