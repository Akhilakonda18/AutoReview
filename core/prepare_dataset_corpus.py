
import json
import re
from pathlib import Path

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "research_dataset"
    / "arxiv-metadata-oai-snapshot.json"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "research_dataset"
    / "processed"
)

CORPUS_PATH = PROCESSED_DIR / "rag_corpus.jsonl"

# Maximum number of selected papers to index.
MAX_PAPERS = 5000

# Keywords used to select research relevant to AutoReview.
KEYWORDS = [
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "large language model",
    "large language models",
    "automated literature review",
    "research paper review",
    "scientific document",
    "scientific literature",
    "citation recommendation",
    "citation analysis",
    "academic writing",
    "paper summarization",
    "factuality",
    "hallucination detection",
    "evidence verification",
    "document retrieval",
    "information retrieval",
    "text similarity",
    "plagiarism detection",
    "research evaluation",
    "scientific peer review",
    "natural language processing",
]


def normalize_text(value):
    """Convert arXiv metadata text into clean searchable text."""
    if not isinstance(value, str):
        return ""

    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def is_relevant(title, abstract):
    """Select papers whose title or abstract matches project topics."""
    combined = f"{title} {abstract}".lower()

    return any(keyword in combined for keyword in KEYWORDS)


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"arXiv metadata file not found: {SOURCE_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing the AutoReview research corpus.")
    print(f"Source: {SOURCE_PATH}")
    print(f"Maximum selected papers: {MAX_PAPERS}")
    print("Reading metadata one record at a time...")

    selected = 0
    scanned = 0

    # Write to a temporary file first so an interrupted run does not
    # leave a partially written corpus at the final path.
    temporary_path = CORPUS_PATH.with_suffix(".jsonl.tmp")

    try:
        with SOURCE_PATH.open(
            "r", encoding="utf-8"
        ) as source, temporary_path.open(
            "w", encoding="utf-8"
        ) as output:

            for line in source:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                scanned += 1

                title = normalize_text(record.get("title", ""))
                abstract = normalize_text(record.get("abstract", ""))

                if not title or not abstract:
                    continue

                if not is_relevant(title, abstract):
                    continue

                paper_id = normalize_text(
                    record.get("id", "")
                )

                categories = record.get("categories", "")
                if isinstance(categories, list):
                    categories = " ".join(
                        str(item) for item in categories
                    )

                document = {
                    "id": paper_id,
                    "title": title,
                    "text": abstract,
                    "source": "arXiv",
                    "categories": normalize_text(
                        str(categories)
                    ),
                }

                output.write(
                    json.dumps(
                        document,
                        ensure_ascii=False
                    ) + "\n"
                )

                selected += 1

                if selected % 500 == 0:
                    print(
                        f"Selected {selected} papers; "
                        f"scanned {scanned} records."
                    )

                if selected >= MAX_PAPERS:
                    break

        if selected == 0:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(
                "No relevant papers were selected. "
                "Check the metadata format and keywords."
            )

        temporary_path.replace(CORPUS_PATH)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print("\nCorpus preparation completed.")
    print(f"Records scanned: {scanned}")
    print(f"Relevant papers selected: {selected}")
    print(f"Corpus saved to: {CORPUS_PATH}")
    print(
        "\nNote: the corpus contains titles and abstracts, "
        "not full research-paper text."
    )


if __name__ == "__main__":
    main()