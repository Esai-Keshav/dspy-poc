import time

start = time.perf_counter()
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="smollm2:360m-instruct-q4_K_M")

response = llm.invoke("What is the capital of France?")
print(response)
end = time.perf_counter()
print(f"Time taken: {end - start:.4f} seconds")
