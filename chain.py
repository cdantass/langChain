from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Specify model llm
llm = ChatOllama(
    model="llama3.1:8b"
)

# Prompt I want to talk about
prompt = ChatPromptTemplate.from_template(
    "Fale o que é {theme} em apenas 1 linha"
)

# Parser
parser = StrOutputParser()

# Prompt → llm → parser
chain = prompt | llm | parser

# Response sent to llm
response = chain.invoke({
    "theme": "RAG no mundo da IA"
})

# Print response
print(response)