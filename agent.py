from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain_experimental.llms.ollama_functions import OllamaFunctions


search = TavilySearchResults()
# rst = search.invoke("what is the weather in SF")
# print(rst)

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
# print(documents)
embeddings = OllamaEmbeddings(model="llama2:13b")
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
# rst2 = retriever.get_relevant_documents("A common workflow")

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)
tools = [search, retriever_tool]

# llm = Ollama(model="llama2:13b")
llm = OllamaFunctions(model="llama2:13b", temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor.invoke({"input": "how can langsmith help with testing?"})

agent_executor.invoke({"input": "what's F1?"})
