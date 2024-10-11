from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"

memory = ConversationBufferMemory(memory_key="chat_history")


template_str = """
You are a chatbot having a conversation with a human.
Previous conversation:
{chat_history}
Human: {question}
AI:"""

prompt_template = PromptTemplate.from_template(template_str)

llm = ChatOpenAI(temperature=0)

memory_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True,
    memory=memory,
)


print(memory_chain.predict(question="你好，我是jack"))
print(memory_chain.predict(question="你还记得我叫什么吗？"))
