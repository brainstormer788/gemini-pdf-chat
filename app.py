import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import base64

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

load_dotenv()


# ---------------------- RAG ENGINE SETUP ----------------------
def initialize_chat_engine(documents):

    llm = Gemini(
        model="models/gemini-1.5-pro",     # ✅ HIGH-QUALITY + AVAILABLE
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5,
    )

    embed = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed

    # ✅ UNIVERSAL HIGH-QUALITY EXPLANATION PROMPT
    Settings.system_prompt = """
You are an adaptive, high-level academic explanation assistant.
You ALWAYS answer strictly using the uploaded PDF. Do not invent information.

Your goals:
- Detect the subject and difficulty level automatically.
- Teach concepts clearly, deeply, and logically.
- Make the user understand, not just memorize.

When answering:
1. Start with a **concise overview** of the requested topic.
2. Then **break the explanation into structured sections** with headings.
3. Explain in **simple language first**, then add **technical depth**.
4. ALWAYS provide:
   - Real-world examples
   - Analogies
   - Industry or everyday applications
   - Tables / comparisons when helpful
5. For large topics (e.g., whole units/chapters):
   - Start with summary → key themes → detailed topic-wise breakdown → final revision notes.

If the answer is NOT in the PDF, reply:
"The PDF does not contain this information."
"""

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",   # ✅ Enables memory across conversation
        similarity_top_k=5
    )


# ---------------------- PDF PREVIEW UI ----------------------
def display_pdf_from_bytes(pdf_bytes):
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.sidebar.subheader("📄 PDF Preview")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True
    )


# ---------------------- MAIN UI ----------------------
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (with Memory & Deep Explanations)")

    # Session State Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # ✅ Rebuild chat engine after rerun if PDF already loaded
    if st.session_state.pdf_bytes is not None and st.session_state.chat_engine is None:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, st.session_state.pdf_name)
        with open(file_path, "wb") as f:
            f.write(st.session_state.pdf_bytes)
        docs = SimpleDirectoryReader(temp_dir).load_data()
        st.session_state.chat_engine = initialize_chat_engine(docs)

    # Sidebar: Upload Section
    with st.sidebar:
        st.subheader("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_pdf is not None:
            if uploaded_pdf.name != st.session_state.pdf_name:
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()

                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, uploaded_pdf.name)
                with open(file_path, "wb") as f:
                    f.write(st.session_state.pdf_bytes)

                docs = SimpleDirectoryReader(temp_dir).load_data()

                st.session_state.chat_engine = initialize_chat_engine(docs)
                st.session_state.messages = []   # Reset chat history
                st.success("✅ PDF Loaded Successfully!")
                st.experimental_rerun()

        if st.session_state.pdf_bytes:
            display_pdf_from_bytes(st.session_state.pdf_bytes)

    # Chat History Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    user_msg = st.chat_input("Ask anything about your PDF...")

    if user_msg:
        if st.session_state.chat_engine is None:
            st.error("⚠️ Please upload a PDF first.")
            return

        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = st.session_state.chat_engine.chat(user_msg).response
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)


if __name__ == "__main__":
    main()
