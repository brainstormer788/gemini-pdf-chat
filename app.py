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
    """Create RAG + Memory chat engine."""
    
    llm = Gemini(
        model="models/gemini-2.0-flash",   # ✅ Works with new Gemini API Keys
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
        "You are a helpful assistant. Answer ONLY using information inside the uploaded PDF. "
        "If the PDF does not contain the answer, say 'The PDF does not contain this information.'"
    )

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5
    )


def display_pdf_from_bytes(pdf_bytes):
    """Preview PDF stored in session."""
    st.sidebar.subheader("📄 PDF Preview")
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (with Memory)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # -------------------- SIDEBAR --------------------
    with st.sidebar:
        st.subheader("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_pdf is not None:
            # Run only first upload, not every rerun
            if "pdf_bytes" not in st.session_state or uploaded_pdf.name != st.session_state.get("pdf_name"):
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()

                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_pdf.name)
                with open(file_path, "wb") as f:
                    f.write(st.session_state.pdf_bytes)

                docs = SimpleDirectoryReader(temp_dir).load_data()
                st.session_state.chat_engine = initialize_chat_engine(docs)
                st.session_state.messages = []  # reset chat history
                st.success("✅ PDF Loaded Successfully!")
                st.experimental_rerun()

        # If PDF already loaded → show preview
        if "pdf_bytes" in st.session_state:
            display_pdf_from_bytes(st.session_state.pdf_bytes)

    # -------------------- CHAT DISPLAY --------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------------------- CHAT INPUT --------------------
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
