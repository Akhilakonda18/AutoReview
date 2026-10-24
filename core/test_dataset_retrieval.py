
import json
import pickle
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "research_dataset"
    / "index"
)

with (INDEX_DIR / "tfidf_vectorizer.pkl").open("rb") as file:
    vectorizer = pickle.load(file)

with (INDEX_DIR / "tfidf_matrix.pkl").open("rb") as file:
    matrix = pickle.load(file)

documents = []

with (INDEX_DIR / "documents.jsonl").open(
    "r", encoding="utf-8"
) as file:
    for line in file:
        if line.strip():
            documents.append(json.loads(line))

query = (
    "How do large language models use retrieval "
    "augmented generation to improve factual accuracy?"
)

query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, matrix).flatten()

top_indices = scores.argsort()[::-1][:5]

print("\nTOP 5 RETRIEVED RESEARCH PAPERS")
print("=" * 60)

for rank, index in enumerate(top_indices, start=1):
    paper = documents[index]

    print(f"\n{rank}. {paper['title']}")
    print(f"arXiv ID: {paper['id']}")
    print(f"Similarity: {scores[index]:.4f}")
    print(f"Source: {paper['source']}")
    print(f"Abstract excerpt: {paper['text'][:350]}...")
