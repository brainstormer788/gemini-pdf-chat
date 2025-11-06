import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import base64

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini           # ✅ Stable / available here
from llama_index.embeddings.gemini import GeminiEmbedding  # ✅ Stable / available here

load_dotenv()

# -----------------------
# Helpers
# -----------------------
def initialize_chat_engine(documents):
    """Build vector index + memory chat engine using Gemini (deprecated wrappers but stable)."""

    # LLM (Gemini 2.x models work when referenced like this in the wrapper)
    llm = Gemini(
        model="models/gemini-2.0-flash",          # ✅ works with your new API key
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    # Embeddings
    embed = GeminiEmbedding(
        model_name="models/text-embedding-004",   # ✅ recommended embedding model
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Apply globally for LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed
    # Do NOT pass system_prompt to condense_question chat engine (it errors). Keep it global only.
    Settings.system_prompt = (
        "You are a helpful assistant. Answer strictly from the uploaded PDF. "
        "If the answer is not in the PDF, say: 'The PDF does not contain this information.'"
    )

    # Build index
    index = VectorStoreIndex.from_documents(documents)

    # Memory-enabled chat engine
    return index.as_chat_engine(
        chat_mode="condense_question",   # ✅ enables conversation memory
        similarity_top_k=5,
    )


def display_pdf(pdf_file):
    """Preview PDF in sidebar."""
    st.sidebar.subheader("📄 PDF Preview")
    encoded = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


# -----------------------
# App
# -----------------------
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (RAG + Memory)")

    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False
    if "loaded_pdf_name" not in st.session_state:
        st.session_state.loaded_pdf_name = None

    # Sidebar: upload
    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Choose a PDF", type="pdf")

        if pdf:
            # Compare by filename to survive Streamlit reruns
            if pdf.name != st.session_state.loaded_pdf_name:
                st.session_state.loaded_pdf_name = pdf.name

                # Save to temp dir
                tmp = tempfile.mkdtemp()
                path = os.path.join(tmp, pdf.name)
                with open(path, "wb") as f:
                    f.write(pdf.getbuffer())

                # Load documents
                documents = SimpleDirectoryReader(tmp).load_data()

                # Build chat engine
                st.session_state.chat_engine = initialize_chat_engine(documents)
                st.session_state.pdf_loaded = True
                st.session_state.messages = []   # reset chat on new PDF

                display_pdf(pdf)
                # Force a rerun to “lock” state in this Streamlit session
                st.experimental_rerun()

        # If already loaded (after rerun), show preview again
        if st.session_state.pdf_loaded and pdf and pdf.name == st.session_state.loaded_pdf_name:
            display_pdf(pdf)

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    user_input = st.chat_input("Ask your PDF a question...")
    if user_input:
        if not st.session_state.pdf_loaded or st.session_state.chat_engine is None:
            st.error("⚠️ Please upload a PDF first.")
            return

        # Show + store user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get model reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = st.session_state.chat_engine.chat(user_input).response
                except Exception as e:
                    # Graceful fallback if anything weird happens
                    reply = f"Sorry, I hit an error while answering: {e}"

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)


if __name__ == "__main__":
    main()
