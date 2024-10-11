from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langserve import add_routes
import os

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = llm | StrOutputParser()

app = FastAPI(title="test")

add_routes(app, rag_chain, path="/test")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
