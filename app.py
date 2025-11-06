import streamlit as st
import os
from dotenv import load_dotenv
import fitz
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------- PDF TEXT LOADER --------
def load_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# -------- TEXT CHUNKER --------
def chunk_text(text, chunk_size=1200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# -------- EMBEDDING --------
def embed_text(text_list):
    embeddings = []
    for t in text_list:
        res = genai.embeddings.embed_content(model="models/text-embedding-004", content=t)
        embeddings.append(res["embedding"])
    return np.array(embeddings)

# -------- RAG SEARCH --------
def retrieve_answer(query, chunks, embeddings):
    query_vec = genai.embeddings.embed_content(model="models/text-embedding-004", content=query)["embedding"]
    sims = cosine_similarity([query_vec], embeddings)[0]
    best_idx = np.argmax(sims)
    return chunks[best_idx]

# -------- MAIN STREAMLIT APP --------
def main():
    st.set_page_config(page_title="Gemini PDF Chatbot")
    st.title("📚 Gemini PDF Chatbot (RAG + Deep Answers)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
        st.session_state.embeddings = None

    with st.sidebar:
        pdf = st.file_uploader("Upload PDF", type="pdf")
        if pdf:
            text = load_pdf_text(pdf.getvalue())
            st.session_state.chunks = chunk_text(text)
            st.session_state.embeddings = embed_text(st.session_state.chunks)
            st.success("✅ PDF Processed Successfully!")

    # Show chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask something about your PDF...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.chunks is None:
                    reply = "⚠️ Please upload a PDF first."
                else:
                    context = retrieve_answer(prompt, st.session_state.chunks, st.session_state.embeddings)
                    model = genai.GenerativeModel("models/gemini-1.5-flash")
                    reply = model.generate_content(
                        f"""
You are a professor-level expert. Answer using ONLY the following PDF content.

PDF CONTENT:
{context}

QUESTION:
{prompt}

Write a clear, structured, deep explanation with examples.
                        """
                    ).text

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)

if __name__ == "__main__":
    main()
