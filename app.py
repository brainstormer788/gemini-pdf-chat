import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import base64

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.google_genai import GoogleLLM
from llama_index.embeddings.google_genai import GoogleTextEmbedding

load_dotenv()


def initialize_chat_engine(documents):
    llm = GoogleLLM(
        model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.4,
    )

    embed = GoogleTextEmbedding(
        model_name="text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed
    Settings.system_prompt = (
        "You are a helpful assistant. Only use information from the PDF. "
        "If the answer is not in the PDF, say: 'The PDF does not contain this information.'"
    )

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",
        similarity_top_k=5
    )


def display_pdf(pdf_file):
    st.sidebar.subheader("📄 PDF Preview")
    encoded = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
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

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF", type="pdf")

        if pdf_file:
            if "loaded_pdf" not in st.session_state or pdf_file.name != st.session_state.loaded_pdf:
                st.session_state.loaded_pdf = pdf_file.name

                tmp = tempfile.mkdtemp()
                path = os.path.join(tmp, pdf_file.name)

                with open(path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                documents = SimpleDirectoryReader(tmp).load_data()

                # Initialize chat model memory-enabled RAG
                st.session_state.chat_engine = initialize_chat_engine(documents)

                display_pdf(pdf_file)
                st.success("✅ PDF Loaded — Ask anything below!")
                st.experimental_rerun()

    # Display past messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat box
    user_input = st.chat_input("Ask your PDF a question...")

    if user_input:
        if st.session_state.chat_engine is None:
            st.error("⚠️ Please upload a PDF first.")
            return

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(user_input).response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)


if __name__ == "__main__":
    main()
