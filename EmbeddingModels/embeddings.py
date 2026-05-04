from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=64,
    )

texts = ["you are going to learn genAI", "genAI is the future of technology", "genAI is the future of technology and it will change the world"]
vectors = embeddings.embed_documents(texts)

# vector = embeddings.embed_query("you are going to learn genAI")
print(vector)