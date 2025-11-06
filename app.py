import streamlit as st
import os
from dotenv import load_dotenv
import fitz
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ✅ Ensure REST transport (prevents NotFound errors)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"), transport="rest")

# -------- PDF TO TEXT --------
def load_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# -------- TEXT CHUNKING --------
def chunk_text(text, chunk_size=1200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -------- EMBEDDING --------
def embed_text(text_list):
    embs = []
    for text in text_list:
        r = genai.embed_content(
            model="text-embedding-004",
            content=text
        )
        embs.append(r["embedding"])
    return np.array(embs)

# -------- RETRIEVAL --------
def retrieve_best_chunk(query, chunks, embeddings):
    q = genai.embed_content(
        model="text-embedding-004",
        content=query
    )["embedding"]
    sims = cosine_similarity([q], embeddings)[0]
    return chunks[np.argmax(sims)]

# -------- MAIN APP --------
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
    st.title("📚 Gemini PDF Chatbot (RAG + Deep Explanations)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
        st.session_state.embeddings = None

    with st.sidebar:
        st.subheader("Upload PDF")
        pdf = st.file_uploader("Choose a PDF", type="pdf")

        if pdf:
            text = load_pdf_text(pdf.getvalue())
            st.session_state.chunks = chunk_text(text)
            st.session_state.embeddings = embed_text(st.session_state.chunks)
            st.success("✅ PDF processed successfully. Ask questions below.")

    # display conversation
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask something about your PDF...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.chunks is None:
                    reply = "⚠️ Please upload a PDF first."
                else:
                    context = retrieve_best_chunk(
                        user_query,
                        st.session_state.chunks,
                        st.session_state.embeddings
                    )

                    model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ final stable name

                    reply = model.generate_content(
                        f"""
You are an expert teacher. Answer using ONLY the PDF content below.

PDF CONTENT:
{context}

QUESTION:
{user_query}

Write a deep, structured explanation with:
- Real-world examples
- Clear bullet points
- Step-by-step logic
"""
                    ).text

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)

if __name__ == "__main__":
    main()
