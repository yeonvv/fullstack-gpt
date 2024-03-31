import streamlit as st

from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask question to an AI about your files!
"""
)


def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file=file_path, mode="wb") as f:
        f.write(file_content)
    embeddings = OpenAIEmbeddings()
    loader = UnstructuredFileLoader(
        f"./.cache/files/{file.name}",
    )
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    docs = loader.load_and_split(text_splitter=splitter)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cache_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


file = st.file_uploader(
    label="Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

question = st.chat_input(placeholder="Ask your question in document")

if file:
    retriever = embed_file(file)
