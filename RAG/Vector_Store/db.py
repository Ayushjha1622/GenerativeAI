from langchain_core.tools import retriever
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma.from_documents(
    documents= docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)

result = vector_store.similarity_search("what is used for data analysis",k=2)

for i in result:
    print(i.page_content)
    print(i.metadata)
   
retriever = vector_store.as_retriever()
docs = retriever.invoke("what is used for data analysis")

for i in docs:
    print(i.page_content)
   