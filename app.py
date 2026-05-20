import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

st.set_page_config(page_title="My AI RAG Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 My AI RAG Chatbot")
st.markdown("**Built by Sahil Shaikh**")

# OpenAI API Key
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

openai_key = st.text_input("Enter OpenAI API Key 🔑", value=st.session_state.openai_key, type="password")

if openai_key:
    st.session_state.openai_key = openai_key
    os.environ["OPENAI_API_KEY"] = openai_key
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

# File Upload
st.subheader("Upload Your Documents")
uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True, type=["pdf", "txt"])

if st.button("Process Documents") and uploaded_files:
    with st.spinner("Processing your documents..."):
        # Save uploaded files
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Load and index documents
        documents = SimpleDirectoryReader(".").load_data()
        index = VectorStoreIndex.from_documents(documents)
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question")
        st.success(f"✅ {len(uploaded_files)} document(s) processed successfully!")

# Chat Interface
if "chat_engine" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if prompt := st.chat_input("Ask questions about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
else:
    st.info("👆 Enter your OpenAI key and upload documents to start chatting!")