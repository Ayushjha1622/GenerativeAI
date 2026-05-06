from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent


search_tool = TavilySearch(max_results = 5)

llm = ChatMistralAI(model="open-mistral-nemo", timeout=120)

prompt = ChatPromptTemplate.from_template(
    """
    You are a news summarizer. Summarize the following news article in simple words.
     {news}
    """
)

chain = prompt | llm | StrOutputParser() 


news = search_tool.run("latest AI news of 2026")

result = chain.invoke({"news" : news})

print(result)