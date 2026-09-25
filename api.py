from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(
    model="qwen3:1.7b"
)

prompt = "Fale sobre o que é LangChain, apenas em 2 linhas"

messages = [
    SystemMessage(
        content="Você é um professor sobre IA/Machine Learning"
    ),
    HumanMessage(
        content=prompt
    )
]

resposta = llm.invoke(messages)

print(resposta.content)