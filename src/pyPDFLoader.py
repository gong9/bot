from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"


loader = PyPDFLoader("test.pdf", extract_images=True)
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)


db = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db"
)
