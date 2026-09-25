from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:1.7b")

pergunta = input("Pergunta: ")

resposta = llm.invoke(pergunta)

print(resposta.content)