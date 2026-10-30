
import streamlit as st
import os

from dotenv import load_dotenv
from google import genai

from rag import (
    extract_text,
    create_chunks,
    build_vector_index,
    retrieve_chunks,
    generate_answer
)


# Load API key
load_dotenv()

# Page settings
st.set_page_config(
    page_title="College Notes AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Powered College Notes & Exam Assistant")
st.caption("Your personal AI study assistant powered by RAG.")

# Connect Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# Store information during the session
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "history" not in st.session_state:
    st.session_state.history = []


# Sidebar
st.sidebar.header("📚 Upload Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload your college notes (PDF)",
    type=["pdf"]
)


# Process uploaded PDF
if uploaded_file:

    if st.sidebar.button("Process PDF"):

        with st.spinner("Reading notes and generating embeddings..."):

            try:
                pages = extract_text(uploaded_file.getvalue())

                if not pages:
                    st.error("No selectable text found in this PDF.")
                    st.stop()

                chunks = create_chunks(pages)

                index = build_vector_index(client, chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.history = []

                st.sidebar.success("PDF processed successfully!")

            except Exception as error:
                st.error(f"Processing failed: {error}")


# Ask questions
if st.session_state.index is not None:

    st.success(
        f"Knowledge base ready: "
        f"{len(st.session_state.chunks)} text chunks indexed."
    )

    question = st.text_input(
        "Ask anything from your uploaded notes",
        placeholder="Example: Explain Banker's Algorithm..."
    )

    if st.button("Ask AI") and question.strip():

        with st.spinner("Searching notes and generating answer..."):

            try:
                results = retrieve_chunks(
                    client,
                    st.session_state.index,
                    st.session_state.chunks,
                    question
                )

                answer = generate_answer(
                    client,
                    question,
                    results
                )

                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "sources": results
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")


    # Display previous questions and answers
    for item in reversed(st.session_state.history):

        st.markdown("---")

        st.markdown("### 🧑‍🎓 Your Question")
        st.write(item["question"])

        st.markdown("### 🤖 AI Answer")
        st.markdown(item["answer"])

        with st.expander("📖 View retrieved source passages"):

            for source in item["sources"]:

                st.write(f"**Page {source['page']}**")
                st.write(source["text"])

                st.caption(
                    f"Similarity score: {source['score']:.3f}"
                )


else:

    st.info("Upload a PDF from the sidebar to get started.")

    st.markdown("""
    ### Features

    - Chat with your college notes
    - Semantic search using AI embeddings
    - Context-based answers
    - Source page references
    """)
