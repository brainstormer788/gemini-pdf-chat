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

    # LLM - You can switch between flash/pro:
    # "models/gemini-2.0-flash" (fast) or "models/gemini-2.0-pro" (deep answers)
    llm = Gemini(
        model="models/gemini-2.0-pro",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5,
    )

    embed = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed

    # ✅ UNIVERSAL INTELLIGENT EXPLANATION PROMPT (Works for ALL Subjects)
    Settings.system_prompt = """
You are an adaptive, high-level academic explanation assistant.
You ALWAYS answer strictly using the uploaded PDF. Do not invent information.

Your goals:
- Understand the subject type from the PDF (engineering, law, medicine, business, science, humanities).
- Adjust explanation depth accordingly.
- Teach concepts clearly and deeply.
- Make the user feel confident enough to explain the concept to someone else.

When answering:
1. Start with a short overview.
2. Break concepts into clear sections with headings.
3. Explain step-by-step in simple language.
4. Then go deeper for understanding.
5. ALWAYS provide real-world examples, analogies, or case applications.
6. Use bullet points, numbered lists, tables, and structured formatting.
7. If asked for full unit/chapter explanation → Summarize first → Then expand topic by topic.

If the requested answer is NOT in the PDF, say:
"The PDF does not contain this information."

Your tone:
- Clear
- Supportive
- Detailed
- Teacher-like
"""

    index = VectorStoreIndex.from_documents(documents)

    return index.as_chat_engine(
        chat_mode="condense_question",   # enables memory-based conversation
        similarity_top_k=5
    )


# ---------------------- PDF PREVIEW UI ----------------------
def display_pdf_from_bytes(pdf_bytes):
    encoded = base64.b64encode(pdf_bytes).decode("utf-8")
    st.sidebar.subheader("📄 PDF Preview")
    st.sidebar.markdown(
        f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="450"></iframe>',
        unsafe_allow_html=True,
    )


# ---------------------- MAIN UI ----------------------
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (with Memory & Deep Explanations)")

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # ✅ Rebuild engine after rerun if PDF already loaded
    if st.session_state.pdf_bytes is not None and st.session_state.chat_engine is None:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, st.session_state.pdf_name)
        with open(path, "wb") as f:
            f.write(st.session_state.pdf_bytes)
        documents = SimpleDirectoryReader(temp_dir).load_data()
        st.session_state.chat_engine = initialize_chat_engine(documents)

    # Sidebar Upload
    with st.sidebar:
        st.subheader("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF", type="pdf")

        if uploaded_pdf is not None:
            if uploaded_pdf.name != st.session_state.pdf_name:
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_bytes = uploaded_pdf.getvalue()

                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, uploaded_pdf.name)
                with open(path, "wb") as f:
                    f.write(st.session_state.pdf_bytes)

                documents = SimpleDirectoryReader(temp_dir).load_data()

                st.session_state.chat_engine = initialize_chat_engine(documents)
                st.session_state.messages = []  # reset chat history
                st.success("✅ PDF Loaded Successfully!")
                st.experimental_rerun()

        if st.session_state.pdf_bytes:
            display_pdf_from_bytes(st.session_state.pdf_bytes)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your PDF...")

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
