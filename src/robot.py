from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import os

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"


prompt_template_str = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""


prompt_template = PromptTemplate.from_template(prompt_template_str)


# 填充context
vectorstore = Chroma(
    persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

app = FastAPI(title="robot")

add_routes(app, rag_chain, path="/robot")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
