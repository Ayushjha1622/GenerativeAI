import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# ------------------ Page Config ------------------
st.set_page_config(page_title="Movie Extractor", page_icon="🎬")

st.title("🎬 Movie Information Extractor")

# ------------------ Model ------------------
model = ChatMistralAI(model="mistral-small")

# ------------------ Schema ------------------
class Movie(BaseModel):
    title: str 
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# ------------------ Prompt ------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
Extract movie details from the given paragraph.

{format_instructions}
"""),
        ("human", "{paragraph}"),
    ]
)

# ------------------ UI Input ------------------
paragraph = st.text_area("Enter your paragraph:", height=200)

# ------------------ Button ------------------
if st.button("Extract"):
    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
    else:
        with st.spinner("Processing..."):
            final_prompt = prompt.invoke({
                "paragraph": paragraph,
                "format_instructions": parser.get_format_instructions()
            })

            response = model.invoke(final_prompt)

            try:
                parsed_output = parser.parse(response.content)
                st.subheader("Extracted Information")
                st.write(parsed_output)
            except Exception as e:
                st.error("Failed to parse response")
                st.text(response.content)