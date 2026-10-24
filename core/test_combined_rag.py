
from core.rag import retrieve_chunks

query = (
    "How can retrieval augmented generation and large language "
    "models improve factual accuracy in scientific research?"
)

print("\nAUTOREVIEW COMBINED RAG TEST")
print("=" * 65)

results = retrieve_chunks(query, top_k=5)

if not results:
    print("No matching evidence found.")
else:
    for rank, item in enumerate(results, start=1):
        print(f"\n{rank}. {item['filename']}")
        print(f"Source type: {item.get('source_type', 'unknown')}")
        print(f"TF-IDF Similarity: {item.get('similarity', 0.0):.4f}")
        print(f"RRF Score: {item.get('rrf_score', 0.0):.6f}")
        print(f"Source: {item.get('source', 'unknown')}")
        print(f"Evidence: {item['text'][:300]}...")

print("\nTest completed.")
