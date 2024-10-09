from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

prompt_template_str = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""


prompt_template = PromptTemplate.from_template(prompt_template_str)


# 填充context
vectorstore = Chroma(persist_directory='../../chroma_db', embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

docs = retriever.invoke("发生争议如何解决？")

print(len(docs))
