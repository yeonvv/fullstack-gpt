import streamlit as st

from time import sleep

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📑",
)

st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(name=role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(
        message=message["message"],
        role=message["role"],
        save=False,
    )

message = st.chat_input(placeholder="To ask for AI")

if message:
    send_message(message=message, role="human")
    sleep(1)
    send_message(f"You said {message}", role="ai")
