from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os

os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"

# 1. 实例化一个 CommaSeparatedListOutputParser对象
output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()

# 2. 创建一个 prompt template，将 output_parser.get_format_instructions()填充进最终的prompt中
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

# 3. 创建一个LLM实例
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 4. 构建链
chain = prompt | model | output_parser

print(chain.invoke({"subject": "彩虹颜色"}))
