from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}"),
    ]
)

llm = Ollama(model="llama2:13b")
chat_model = ChatOllama()

output_parser = StrOutputParser()

chain1 = prompt | llm | output_parser
chain2 = prompt | chat_model | output_parser

rst = chain2.invoke({"input": "how can langsmith help with testing?"})
print(rst)
