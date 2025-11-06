import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import tempfile
import base64

load_dotenv()

def initialize_chat_engine(documents):
    """Initialize vector index + memory chat engine."""
    
    # LLM (Chat Model)
    llm = Gemini(
        model="models/gemini-2.0-flash",   # ✅ Correct model for new Gemini keys
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    # Embeddings
    embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",  # ✅ Best available embedding model
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Apply settings globally in LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.system_prompt = "You are a helpful assistant. Always answer using ONLY the contents of the uploaded PDF."

    # Build vector index
    index = VectorStoreIndex.from_documents(documents)

    # Create chat engine (RAG + Memory)
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",  # ✅ Includes conversation memory
        similarity_top_k=5,
    )

    return chat_engine


def display_pdf_preview(pdf_file):
    """Show PDF preview inside sidebar."""
    st.sidebar.subheader("PDF Preview")
    encoded = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Gemini RAG Chat", layout="wide")
    st.title("📚 Gemini PDF Chatbot with Memory")

    # Persist chat & engine across reruns
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF", type="pdf")

        if pdf_file:
            if "current_pdf_name" not in st.session_state or pdf_file.name != st.session_state.current_pdf_name:

                st.session_state.current_pdf_name = pdf_file.name

                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, pdf_file.name)

                with open(file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                docs = SimpleDirectoryReader(temp_dir).load_data()

                # Initialize new chat engine
                st.session_state.chat_engine = initialize_chat_engine(docs)

                display_pdf_preview(pdf_file)
                st.success("✅ PDF Loaded Successfully! You can now start chatting.")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat box
    user_input = st.chat_input("Ask your PDF something...")

    if user_input:
        if st.session_state.chat_engine is None:
            st.error("📄 Please upload a PDF first.")
            return

        # Record user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(user_input).response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)


if __name__ == "__main__":
    main()
