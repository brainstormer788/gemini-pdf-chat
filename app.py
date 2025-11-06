import streamlit as st
import os
from dotenv import load_dotenv
import fitz
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# -------- EMBEDDING --------
def embed_text(text_list):
    embeddings = []
    for t in text_list:
        res = genai.embed_content(
            model="text-embedding-004",   # ✅ Correct model name
            content=t
        )
        embeddings.append(res["embedding"])
    return np.array(embeddings)

# -------- RAG SEARCH --------
def retrieve_best_chunk(query, chunks, embeddings):
    query_embedding = genai.embed_content(
        model="text-embedding-004",       # ✅ Correct model name
        content=query
    )["embedding"]

    sims = cosine_similarity([query_embedding], embeddings)[0]
    best_index = np.argmax(sims)
    return chunks[best_index]

# -------- MAIN STREAMLIT APP --------
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
            st.success("✅ PDF processed successfully! Ask your questions below.")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                    context = retrieve_best_chunk(user_query, st.session_state.chunks, st.session_state.embeddings)

                    model = genai.GenerativeModel("gemini-1.5-flash")  # ✅ Correct model

                    prompt = f"""
You are an expert teacher. Answer ONLY using the PDF content.

PDF EXCERPT:
{context}

QUESTION:
{user_query}

Write a highly structured, clear explanation. Include:
- bullet points
- real life examples
- analogies
- simple language
"""
                    reply = model.generate_content(prompt).text

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)

if __name__ == "__main__":
    main()
