from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest")

response = llm.invoke("In one sentence, what is item 1A 10K SEC-fillings?")

print(response)