
# AI-Powered College Notes & Exam Assistant Using RAG

An AI-powered study assistant that allows students to upload academic PDF notes and ask questions based on their study material.

## Features

- Upload academic PDF notes
- Extract text from PDF documents
- Split documents into smaller text chunks
- Generate semantic embeddings using Google Gemini
- Store and search vectors using FAISS
- Retrieve relevant passages using similarity search
- Generate context-based answers using Gemini AI
- Display source page references
- Interactive web interface using Streamlit

## Technology Stack

- Python
- Streamlit
- Google Gemini API
- Gemini Embedding Model
- FAISS Vector Database
- PyMuPDF
- NumPy

## How It Works

1. Student uploads a PDF.
2. Text is extracted from the PDF.
3. Extracted text is divided into chunks.
4. Gemini generates vector embeddings.
5. FAISS stores the embeddings and performs similarity search.
6. Relevant passages are retrieved for the student's question.
7. Gemini generates an answer using the retrieved context.

## Installation

1. Clone the repository.
2. Create a Python virtual environment.
3. Install the required libraries:

   pip install -r requirements.txt

4. Create a `.env` file in the project folder.
5. Add your Gemini API key:

   GEMINI_API_KEY=your_api_key_here

6. Run the application:

   python -m streamlit run app.py

## Security

Never upload your Gemini API key or `.env` file to GitHub.

## Project Type

Academic Project – Artificial Intelligence and Machine Learning
