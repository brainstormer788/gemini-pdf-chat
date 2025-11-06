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

    llm = Gemini(
        model="models/gemini-1.5-flash",   # ✅ Your key supports this model
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.9,   # ✅ Higher creativity = deeper explanations
    )

    embed = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GEMINI_API_KEY")
    )

    Settings.llm = llm
    Settings.embed_model = embed

    # ✅ Strong universal explanation prompt
    Settings.system_prompt = """
You are an advanced academic explanation assistant.
You ALWAYS answer using only the content of the uploaded PDF.

When answering:
- Start with a short clear summary.
- Then break concepts into sections with headings.
- Explain step-by-step, from simple → deeper understanding.
- Always include:
  * Real-world examples
  * Daily-life analogies
  * Industry or practical applications
  * Tables / comparisons when helpful
- If user asks for entire chapter/unit, summarize first → then detailed topic-by-topic notes.
- Use bullet points, numbered lists, formatting.

If answer is not in the PDF, respond:
"The PDF does not contain this information."
"""

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
    st.title("📚 Gemini PDF Chatbot (Deep Understanding)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_name = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # Rebuild engine after refresh if PDF exists
    if st.session_state.pdf_bytes is not None and st.session_state.chat_engine is None:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, st.session_state.pdf_name)
        with open(path, "wb") as f:
            f.write(st.session_state.pdf_bytes)
        docs = SimpleDirectoryReader(tmp).load_data()
        st.session_state.chat_engine = initialize_chat_engine(docs)

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Choose a PDF", type="pdf")

        if pdf is not None and pdf.name != st.session_state.pdf_name:
            st.session_state.pdf_name = pdf.name
            st.session_state.pdf_bytes = pdf.getvalue()

            tmp = tempfile.mkdtemp()
            path = os.path.join(tmp, pdf.name)
            with open(path, "wb") as f:
                f.write(st.session_state.pdf_bytes)

            docs = SimpleDirectoryReader(tmp).load_data()
            st.session_state.chat_engine = initialize_chat_engine(docs)
            st.session_state.messages = []
            st.success("✅ PDF Loaded Successfully!")
            st.experimental_rerun()

        if st.session_state.pdf_bytes:
            display_pdf_from_bytes(st.session_state.pdf_bytes)

    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask anything about your PDF...")

    if prompt:
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
