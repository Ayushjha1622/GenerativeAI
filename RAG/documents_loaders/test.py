from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

data = TextLoader("documents_loaders/notes.txt")

loader = data.load()

chunks = splitter.split_documents(loader)
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()

