from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:latest")

response = llm.invoke("who is godwin addetsi?")

print(response)