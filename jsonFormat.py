from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class IaAtual(BaseModel):
    tema: str = Field(
        description="Tema analisado"
    )
    explicacao: str = Field(
        description="Explicação em apenas 1 linha sobre como o tema está atualmente"
    )


class IaAntigamente(BaseModel):
    tema: str = Field(
        description="Tema analisado"
    )
    explicacao: str = Field(
        description="Explicação em apenas 1 linha sobre como o tema era antigamente"
    )


class Comparacao(BaseModel):
    tema: str = Field(
        description="Tema analisado"
    )
    explicacao: str = Field(
        description="Principais mudanças entre antigamente e atualmente"
    )


llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


llm_atual = llm.with_structured_output(IaAtual)
llm_antigo = llm.with_structured_output(IaAntigamente)
llm_comparacao = llm.with_structured_output(Comparacao)


prompt_atual = ChatPromptTemplate.from_template(
    "Explique em uma linha como {tema} é atualmente."
)

prompt_antigo = ChatPromptTemplate.from_template(
    "Explique em uma linha como {tema} era antigamente."
)

prompt_comparacao = ChatPromptTemplate.from_template("""
Você receberá duas descrições sobre o tema {tema}.

DESCRIÇÃO ANTIGA:
{antigo}

DESCRIÇÃO ATUAL:
{atual}

Sua tarefa é COMPARAR as duas descrições.

Identifique especificamente:
- o que mudou;
- qual foi a principal evolução.

""")


chain_atual = prompt_atual | llm_atual

chain_antigo = prompt_antigo | llm_antigo

chain_comparacao = prompt_comparacao | llm_comparacao


tema = "Machine Learning"

atual = chain_atual.invoke({
    "tema": tema
})

antigo = chain_antigo.invoke({
    "tema": tema
})

comparacao = chain_comparacao.invoke({
    "tema": tema,
    "atual": atual.explicacao,
    "antigo": antigo.explicacao
})


print("ATUAL:")
print(atual)

print("\nANTIGAMENTE:")
print(antigo)

print("\nCOMPARAÇÃO:")
print(comparacao)