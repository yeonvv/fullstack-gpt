import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)

st.title("DocumentGPT")


with st.sidebar:
    file = st.file_uploader(
        label="Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


@st.cache_data(show_spinner="Embedding file...")
# file argument가 동일하다면 streamlit은 이 함수를 재실행하지 않는다
# 실행하는 대신 기존에 반환했던 값을 다시 반환한다
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file=file_path, mode="wb") as f:
        f.write(file_content)
    embeddings = OpenAIEmbeddings()
    loader = UnstructuredFileLoader(file_path)
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


llm = ChatOpenAI(temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON't make the anything up.
            
            Context: {context}
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def send_message(msg, role, save=True):
    with st.chat_message(role):
        st.markdown(msg)
    if save:
        st.session_state["messages"].append({"msg": msg, "role": role})


def paint_msg():
    for msg in st.session_state["messages"]:
        send_message(msg=msg["msg"], role=msg["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


if file:
    retriever = embed_file(file)
    send_message(msg="I'm ready! Ask away!", role="ai", save=False)
    paint_msg()
    question = st.chat_input(placeholder="Ask your question in document")
    if question:
        send_message(msg=question, role="human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(question)
        send_message(msg=response.content, role="ai")


else:
    st.markdown(
        """
    Welcome!

    Use this chatbot to ask question to an AI about your files!

    Upload your files on the sidebar.
"""
    )
    st.session_state["messages"] = []
