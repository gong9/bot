
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import os

os.environ['OPENAI_API_BASE'] = 'https://api.chatanywhere.com.cn/v1'

loader = WebBaseLoader(
    web_path="https://www.npmjs.com/package/@anov/gis",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        ["p",]
    ))
)
# 文档加载
docs = loader.load()

print(docs)

# 文本切割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 向量化存储
db = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db")
