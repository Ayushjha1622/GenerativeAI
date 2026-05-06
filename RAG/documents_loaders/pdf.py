from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("documents_loaders/GRU.pdf")

docs = data.load()


splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=100,
    chunk_overlap=10,
)

chunks = splitter.split_documents(docs)

print(len(chunks))