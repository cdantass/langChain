from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="llama3.1:8b"
)

historico = InMemoryChatMessageHistory()

historico.add_user_message("Meu nome é Cauã")
historico.add_ai_message("Prazer em conhecê-lo")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de viagens."),
    MessagesPlaceholder("chat_history"),
    ("human", "{pergunta}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "chat_history": historico.messages,
    "pergunta": "Qual é meu nome?"
})

print(response)