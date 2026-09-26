from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Modelo
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente útil."
    ),
    MessagesPlaceholder(
        variable_name="history"
    ),
    (
        "human",
        "{pergunta}"
    )
])

# Chain normal
chain = prompt | llm | StrOutputParser()

# Memória
historico = InMemoryChatMessageHistory()

# Função que retorna o histórico
def get_chat_history(session_id: str):
    return historico

# Adiciona memória à chain
chain_com_memoria = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="pergunta",
    history_messages_key="history"
)

# Primeira pergunta
resposta1 = chain_com_memoria.invoke(
    {
        "pergunta": "Meu nome é Cauã"
    },
    config={
        "configurable": {
            "session_id": "usuario_1"
        }
    }
)

print("IA:", resposta1)

# Segunda pergunta
resposta2 = chain_com_memoria.invoke(
    {
        "pergunta": "Qual é meu nome?"
    },
    config={
        "configurable": {
            "session_id": "usuario_1"
        }
    }
)

print("IA:", resposta2)