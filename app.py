import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import base64

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

load_dotenv()


def initialize_chat_engine(documents):
    """Create RAG + memory chat engine."""

    llm = Gemini(
        model="models/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    embed = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed
    Settings.system_prompt = (
        "You are a helpful assistant. Answer strictly using the uploaded PDF. "
        "If the PDF does not contain the answer, reply: 'The PDF does not contain this information.'"
    )

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5
    )


def display_pdf_from_bytes(pdf_bytes):
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.sidebar.subheader("📄 PDF Preview")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (with Memory)")

    # ----- SESSION STATE -----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # ✅ If PDF already loaded, rebuild chat engine after rerun
    if st.session_state.pdf_bytes is not None and st.session_state.chat_engine is None:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, st.session_state.pdf_name)
        with open(path, "wb") as f:
            f.write(st.session_state.pdf_bytes)
        documents = SimpleDirectoryReader(temp_dir).load_data()
        st.session_state.chat_engine = initialize_chat_engine(documents)

    # ----- SIDEBAR -----
    with st.sidebar:
        st.subheader("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_pdf is not None:
            # Load new PDF only once
            if uploaded_pdf.name != st.session_state.pdf_name:
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()

                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, uploaded_pdf.name)
                with open(path, "wb") as f:
                    f.write(st.session_state.pdf_bytes)

                documents = SimpleDirectoryReader(temp_dir).load_data()

                st.session_state.chat_engine = initialize_chat_engine(documents)
                st.session_state.messages = []  # Reset chat history
                st.success("✅ PDF Loaded Successfully!")
                st.experimental_rerun()

        # Show preview if PDF exists
        if st.session_state.pdf_bytes:
            display_pdf_from_bytes(st.session_state.pdf_bytes)

    # ----- CHAT HISTORY -----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----- CHAT INPUT -----
    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        if st.session_state.chat_engine is None:
            st.error("⚠️ Please upload a PDF first.")
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.session_state.chat_engine.chat(user_input).response
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)


if __name__ == "__main__":
    main()
