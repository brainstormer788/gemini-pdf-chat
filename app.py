import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import base64

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.google import GoogleLLM
from llama_index.embeddings.google import GoogleTextEmbedding

load_dotenv()


def initialize_chat_engine(documents):
    """Initialize vector index + memory chat engine."""

    # LLM (Chat Model)
    llm = GoogleLLM(
        model="gemini-2.0-flash",  # ✅ correct new Gemini model name
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    # Embeddings
    embed = GoogleTextEmbedding(
        model_name="text-embedding-004",  # ✅ correct embedding model
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Apply globally in LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed
    Settings.system_prompt = (
        "You are a helpful assistant. Only use information from the PDF. "
        "If answer is not found in the PDF, say: 'The PDF does not contain this information.'"
    )

    # Build vector index
    index = VectorStoreIndex.from_documents(documents)

    # Create memory-aware chat engine
    return index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5,
    )


def display_pdf(pdf_file):
    """Display PDF preview in the sidebar."""
    st.sidebar.subheader("📄 PDF Preview")
    encoded = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (with Memory)")

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # Sidebar Upload Section
    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Choose a PDF", type="pdf")

        if pdf:
            if "loaded_pdf_name" not in st.session_state or pdf.name != st.session_state.loaded_pdf_name:

                st.session_state.loaded_pdf_name = pdf.name

                # Save to temp folder
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, pdf.name)

                with open(path, "wb") as f:
                    f.write(pdf.getbuffer())

                # Load text
                documents = SimpleDirectoryReader(temp_dir).load_data()

                # Create chat engine
                st.session_state.chat_engine = initialize_chat_engine(documents)

                # Show preview
                display_pdf(pdf)

                st.success("✅ PDF Loaded Successfully! Ask questions below 👇")
                st.experimental_rerun()  # ✅ ensures engine persists

    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat box
    prompt = st.chat_input("Ask something about your PDF...")

    if prompt:
        if st.session_state.chat_engine is None:
            st.error("⚠️ Please upload a PDF first.")
            return

        # Store and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.session_state.chat_engine.chat(prompt).response
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)


if __name__ == "__main__":
    main()
