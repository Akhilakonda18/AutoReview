
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "research_dataset"
    / "processed"
    / "rag_corpus.jsonl"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "research_dataset"
    / "index"
)

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD TRAINING PAPERS
# -------------------------------------------------

def load_corpus(path):
    documents = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            text = (
                str(record.get("title", ""))
                + "\n"
                + str(record.get("text", ""))
            ).strip()

            if text:
                documents.append({
                    "id": record.get("id", ""),
                    "title": record.get("title", ""),
                    "text": text,
                    "source": record.get("source", ""),
                })

    return documents


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    print("Loading training-paper corpus...")
    documents = load_corpus(CORPUS_PATH)

    if not documents:
        raise ValueError("No documents found in the corpus.")

    print(f"Documents loaded: {len(documents)}")
    print("Building TF-IDF index...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype="float32",
    )

    matrix = vectorizer.fit_transform(
        [doc["text"] for doc in documents]
    )

    # Save the searchable matrix, vectorizer and metadata.
    with (INDEX_DIR / "tfidf_vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)

    with (INDEX_DIR / "tfidf_matrix.pkl").open("wb") as file:
        pickle.dump(matrix, file)

    with (INDEX_DIR / "documents.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for doc in documents:
            file.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )

    print("\nRAG index built successfully!")
    print(f"Documents: {matrix.shape[0]}")
    print(f"Features: {matrix.shape[1]}")
    print(f"Saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
