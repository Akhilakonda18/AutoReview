
from core.pdf_parser import extract_text
from core.section_identifier import identify_sections
from core.review_pipeline import (
    build_agent_query,
    AGENT_QUERIES,
    PDF_PATH,
)
from core.rag import retrieve_chunks


def main():
    print("=" * 70)
    print("AUTOREVIEW - RETRIEVAL QUALITY TEST")
    print("=" * 70)

    # Step 1: Extract the research paper
    print("\nStep 1: Extracting research paper...")

    paper_text = extract_text(PDF_PATH)

    if not paper_text or not paper_text.strip():
        print("ERROR: No text extracted from the research paper.")
        return

    print(f"Extracted characters: {len(paper_text):,}")

    # Step 2: Identify sections
    print("\nStep 2: Identifying paper sections...")

    sections = identify_sections(paper_text)

    if not isinstance(sections, dict):
        print("ERROR: Section identification did not return a dictionary.")
        return

    print(f"Sections detected: {len(sections)}")

    # Step 3: Test retrieval for every specialist agent
    print("\nStep 3: Testing agent-specific retrieval...")

    for agent_name in AGENT_QUERIES:
        print("\n" + "=" * 70)
        print(f"AGENT: {agent_name}")
        print("=" * 70)

        query = build_agent_query(
            agent_name,
            paper_text,
            sections,
        )

        print("\nRetrieval query:")
        print(query[:500])

        try:
            results = retrieve_chunks(query, top_k=5)

            if not results:
                print("\nNo evidence retrieved.")
                continue

            print(f"\nRetrieved evidence chunks: {len(results)}")

            for rank, result in enumerate(results, start=1):
                title = result.get(
                    "filename",
                    "Unknown source",
                )

                similarity = result.get("similarity", 0.0)
                rrf_score = result.get("rrf_score", 0.0)
                content = result.get("text", "")

                print(f"\nEvidence {rank}")
                print(f"Title: {title}")
                print(f"Similarity: {similarity:.4f}")
                print(f"RRF score: {rrf_score:.6f}")
                print("Preview:")
                print(content[:300].replace("\n", " "))

        except Exception as error:
            print(f"\nRetrieval error: {error}")

    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
