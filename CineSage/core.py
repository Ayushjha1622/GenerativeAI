from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

model = ChatMistralAI(model="mistral-small")
class Movie(BaseModel):
    title: str 
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
   Extract movie details from the given paragraph
   {format_instructions}
   """),
        ("human", "{paragraph}"),
    ]
)

para = input("give your paragraph: ")
final_prompt = prompt.invoke(
    {"paragraph": para, 
    "format_instructions": parser.get_format_instructions()}
)

reponse = model.invoke(final_prompt)
movie_data = parser.parse(reponse.content)

print(movie_data)
