import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.google_genai import GoogleLLM
from llama_index.embeddings.google_genai import GoogleTextEmbedding
from dotenv import load_dotenv
import tempfile
import base64

load_dotenv()

def initialize_chat_engine(documents):
    """Initialize RAG + Memory Chat Engine using NEW Google GenAI SDK."""

    llm = GoogleLLM(
        model="gemini-2.0-flash",      # ✅ New Google model naming (no "models/")
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    embed_model = GoogleTextEmbedding(
        model_name="text-embedding-004",   # ✅ New embedding name
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.system_prompt = (
        "You are a helpful assistant. Respond ONLY using information from the uploaded PDF. "
        "If the PDF does not contain the answer, say 'The PDF does not contain this information.'"
    )

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5,
    )


def display_pdf_preview(pdf_file):
    st.sidebar.subheader("PDF Preview")
    encoded = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Gemini RAG Chat", layout="wide")
    st.title("📚 Gemini PDF Chatbot with Memory")

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
                path = os.path.join(temp_dir, pdf_file.name)
                with open(path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                docs = SimpleDirectoryReader(temp_dir).load_data()

                st.session_state.chat_engine = initialize_chat_engine(docs)
                display_pdf_preview(pdf_file)
                st.success("✅ PDF Loaded. Start chatting below ↓")

    # Display chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    prompt = st.chat_input("Ask your PDF something...")
    if prompt:
        if st.session_state.chat_engine is None:
            st.error("📄 Please upload a PDF first.")
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
