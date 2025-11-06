import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import tempfile
import shutil
import base64

load_dotenv()

def initialize_chat_engine(documents, embedding_model, generative_model):
    llm = Gemini(
        model=generative_model,
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    embed_model = GeminiEmbedding(
        model_name=embedding_model,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5,
        system_prompt="You are a helpful assistant. Answer only from the PDF."
    )

    return chat_engine

def display_pdf_preview(pdf_file):
    st.sidebar.subheader("PDF Preview")
    base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>'
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Gemini RAG Chat", layout="wide")
    st.title("📚 Gemini PDF Chatbot with Memory")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF", type="pdf")

        if pdf_file:
            if pdf_file != st.session_state.get("current_pdf", None):
                st.session_state.current_pdf = pdf_file

                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, pdf_file.name)
                with open(file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                docs = SimpleDirectoryReader(temp_dir).load_data()
                st.session_state.docs_loaded = True
                st.session_state.documents = docs
                st.session_state.chat_engine = initialize_chat_engine(
                    docs,
                    "models/embedding-001",
                    "gemini-1.5-flash"
                )

                display_pdf_preview(pdf_file)
                st.success("✅ PDF Loaded & Memory Chat Ready!")

    # Show chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your PDF..."):
        if not st.session_state.docs_loaded:
            st.error("Upload a PDF first!")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.session_state.chat_engine.chat(prompt).response
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)

if __name__ == "__main__":
    main()
