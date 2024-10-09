
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path="https://www.gov.cn/jrzg/2013-10/25/content_2515601.htm",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
            class_=("p1")
        ))
)
docs = loader.load()

print(docs)