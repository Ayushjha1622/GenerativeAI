import streamlit as st
import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env variables
load_dotenv()

st.set_page_config(page_title="AI RAG Assistant", layout="wide")

# ---------- UI HEADER ----------
st.title("📚 AI Document Assistant")
st.markdown("Upload a PDF and ask questions based on its content.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    process_btn = st.button("Process Document")

# ---------- SESSION STATE ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- EMBEDDING ----------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# ---------- PROCESS PDF ----------
if process_btn and uploaded_file:
    with st.spinner("Processing document..."):
        temp_path = f"temp_{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20
        )

        chunks = splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma-db"
        )

        st.session_state.vector_store = vector_store

        os.remove(temp_path)

    st.success("✅ Document processed successfully!")

# ---------- LLM ----------
llm = ChatMistralAI(model="mistral-small-latest")

# ---------- PROMPT ----------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        ("human",
         """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# ---------- CHAT UI ----------
st.subheader("💬 Chat")

user_input = st.chat_input("Ask a question about your document...")

if user_input:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF first.")
    else:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(user_input)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({
            "context": context,
            "question": user_input
        })

        response = llm.invoke(final_prompt)

        # Store chat
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", response.content))

# ---------- DISPLAY CHAT ----------
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)