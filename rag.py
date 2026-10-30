
import fitz
import faiss
import numpy as np
from google import genai
from google.genai import types


EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"


# 1. Extract text from PDF
def extract_text(pdf_bytes):
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()
    return pages


# 2. Split notes into smaller chunks
def create_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


# 3. Convert text into AI embeddings
def get_embeddings(client, texts, task_type):
    all_vectors = []

    for start in range(0, len(texts), 50):
        batch = texts[start:start + 50]

        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type
            )
        )

        all_vectors.extend(
            [item.values for item in response.embeddings]
        )

    vectors = np.array(all_vectors, dtype="float32")
    faiss.normalize_L2(vectors)

    return vectors


# 4. Create FAISS vector database
def build_vector_index(client, chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectors = get_embeddings(
        client,
        texts,
        "RETRIEVAL_DOCUMENT"
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


# 5. Find relevant notes for a question
def retrieve_chunks(client, index, chunks, question, top_k=4):
    query_vector = get_embeddings(
        client,
        [question],
        "RETRIEVAL_QUERY"
    )

    scores, indices = index.search(query_vector, top_k)

    results = []

    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            })

    return results


# 6. Generate answer using Gemini AI

# 6. Generate answer using Gemini AI
def generate_answer(client, question, retrieved_chunks):

    if not retrieved_chunks:
        return "I could not find relevant information in your uploaded notes."

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in retrieved_chunks
    )

    prompt = f"""
You are an AI College Notes and Exam Assistant.

Answer the student's question using ONLY the provided academic notes.

Rules:
1. Explain concepts in simple student-friendly language.
2. Use headings and examples where appropriate.
3. Never invent information absent from the notes.
4. Mention the source page numbers.
5. If the answer is not in the notes, clearly say so.

ACADEMIC NOTES:
{context}

STUDENT QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text
