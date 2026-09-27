from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

import asyncio


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.5
)

prompt_consultor_praias = ChatPromptTemplate.from_messages([
    (
        "system",
        "Apresente-se como Sra Praia. Você é especialista nas praias mais seguras do Brasil."
    ),
    ("human", "{query}")
])

prompt_consultor_tecnologia = ChatPromptTemplate.from_messages([
    (
        "system",
        "Apresente-se como Sr Tec. Você é especialista nas cidades mais tecnológicas do Brasil."
    ),
    ("human", "{query}")
])

prompt_roteador = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Classifique a pergunta.

        Responda SOMENTE:

        praia

        ou

        tecnologia
        """
    ),
    ("human", "{query}")
])

chain_praia = (
    prompt_consultor_praias
    | llm
    | StrOutputParser()
)

chain_tecnologia = (
    prompt_consultor_tecnologia
    | llm
    | StrOutputParser()
)

roteador = (
    prompt_roteador
    | llm
    | StrOutputParser()
)

class Estado(BaseModel):
    query: str
    destino: str = ""
    resposta: str = ""

async def no_roteador(
    estado: Estado,
    config: RunnableConfig
):

    destino = await roteador.ainvoke(
        {"query": estado.query},
        config
    )

    return {
        "destino": destino.strip().lower()
    }


async def no_praia(
    estado: Estado,
    config: RunnableConfig
):

    resposta = await chain_praia.ainvoke(
        {"query": estado.query},
        config
    )

    return {
        "resposta": resposta
    }


async def no_tecnologia(
    estado: Estado,
    config: RunnableConfig
):

    resposta = await chain_tecnologia.ainvoke(
        {"query": estado.query},
        config
    )

    return {
        "resposta": resposta
    }

def escolha_no(estado: Estado):

    if estado.destino == "praia":
        return "praia"

    return "tecnologia"

builder = StateGraph(Estado)

builder.add_node(
    "roteador",
    no_roteador
)

builder.add_node(
    "praia",
    no_praia
)

builder.add_node(
    "tecnologia",
    no_tecnologia
)

builder.add_edge(
    START,
    "roteador"
)

builder.add_conditional_edges(
    "roteador",
    escolha_no,
    {
        "praia": "praia",
        "tecnologia": "tecnologia"
    }
)

builder.add_edge(
    "praia",
    END
)

builder.add_edge(
    "tecnologia",
    END
)

graph = builder.compile()

async def main():

    resultado = await graph.ainvoke(
        Estado(
            query="Quero relaxar em um lugar bonito, de frente para o mar"
        )
    )

    print(resultado["resposta"])

asyncio.run(main())