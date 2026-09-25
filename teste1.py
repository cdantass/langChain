from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:1.7b")

prompt = "1 + 1"

response = llm.invoke(prompt)

print(response.content)