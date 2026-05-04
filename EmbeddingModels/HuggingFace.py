from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "you are going to learn genAI",
    "genAI is the future of technology",
    "genAI is the future of technology and it will change the world"
]

vectors = embeddings.embed_documents(texts)

print(vectors[0][:5])  # preview first vector