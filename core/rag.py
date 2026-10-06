import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.pdf_parser import extract_text


REFERENCE_FOLDER = "data/reference papers"


def split_text(
    text,
    chunk_size=1200,
    overlap=200
):
    """
    Split text into overlapping chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_reference_chunks():

    chunks = []

    if not os.path.exists(
        REFERENCE_FOLDER
    ):

        print(
            "Reference paper folder not found."
        )

        return chunks

    for filename in os.listdir(
        REFERENCE_FOLDER
    ):

        if filename.lower().endswith(".pdf"):

            path = os.path.join(
                REFERENCE_FOLDER,
                filename
            )

            print(
                f"Processing: {filename}"
            )

            try:

                text = extract_text(path)

                paper_chunks = split_text(
                    text
                )

                for i, chunk in enumerate(
                    paper_chunks
                ):

                    chunks.append({
                        "filename": filename,
                        "chunk_id": i + 1,
                        "text": chunk
                    })

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )

    return chunks


def retrieve_chunks(
    query,
    top_k=5
):

    chunks = load_reference_chunks()

    if not chunks:
        return []

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    document_vectors = vectorizer.fit_transform(
        documents
    )

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in ranked_indices:

        result = chunks[index].copy()

        result["similarity"] = float(
            similarities[index]
        )

        results.append(result)

    return results