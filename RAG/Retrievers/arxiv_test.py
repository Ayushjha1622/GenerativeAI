from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    query="Attention Is All You Need",
    load_max_docs=2,
    doc_content_chars_max=1000,
)

docs = retriever.invoke("large language models")

for i in docs:
    print(i.page_content)
    print(i.metadata)
    print("-" * 100)




