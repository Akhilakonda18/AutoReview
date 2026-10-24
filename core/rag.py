
import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.pdf_parser import extract_text


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_FOLDER = PROJECT_ROOT / "data" / "reference papers"
ARXIV_INDEX_FOLDER = (
    PROJECT_ROOT / "data" / "research_dataset" / "index"
)

USE_REFERENCE_PDFS = True
USE_ARXIV_DATASET = True

MIN_SIMILARITY = 0.02
CHUNK_SIZE = 450
CHUNK_OVERLAP = 70

CANDIDATE_MULTIPLIER = 6
MIN_CANDIDATES = 30
MAX_QUERY_CHARS = 2500

# Strong topic indicators: automated scientific paper review.
CORE_PHRASES = (
    "automated peer review",
    "automatic peer review",
    "scientific peer review",
    "automated scientific review",
    "automated paper review",
    "research paper review",
    "manuscript review",
    "reviewer feedback",
    "review generation",
    "automated literature review",
    "systematic literature review",
    "literature review",
    "paper assessment",
    "research paper assessment",
    "llm reviewer",
    "llm-based review",
    "multi-agent review",
    "review quality",
    "reviewing scientific papers",
    "scientific manuscript",
    "peer-reviewing",
)

# These terms matter most when they appear together.
REVIEW_TERMS = {
    "review", "reviews", "reviewer", "reviewers", "reviewing",
    "critique", "critiques", "feedback", "manuscript",
    "manuscripts", "peer", "referee", "referees",
    "literature", "assessment", "assessment", "grading",
}

AI_TERMS = {
    "llm", "llms", "language", "transformer", "transformers",
    "generative", "generation", "rag", "retrieval",
    "agent", "agents", "automated", "automation",
    "artificial", "intelligence",
}

METHOD_TERMS = {
    "evidence", "grounded", "hallucination", "faithfulness",
    "factuality", "citation", "citations", "evaluation",
    "quality", "consistency", "reasoning", "verification",
    "relevance", "reliability",
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text):
    """Normalize text and remove control characters."""
    if text is None:
        return ""

    text = str(text).replace("\x00", " ")
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def tokenize(text):
    """Return lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", clean_text(text).lower())


def prepare_query(query):
    """Prepare the query without adding the source paper abstract."""
    return clean_text(query)[:MAX_QUERY_CHARS]


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    text = clean_text(text)

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    step = chunk_size - overlap

    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]


def looks_like_bibliography(text):
    """Identify text dominated by reference-list entries."""
    text = clean_text(text)

    if not text:
        return True

    url_count = len(
        re.findall(r"https?://|www\.", text, flags=re.IGNORECASE)
    )
    year_count = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    citation_count = len(re.findall(r"\[\s*\d{1,3}\s*\]", text))

    return url_count >= 5 or (
        year_count >= 8 and citation_count >= 3
    )


# ============================================================
# DOMAIN RELEVANCE
# ============================================================

def calculate_domain_relevance(item):
    """
    Score relevance to automated scientific paper review.

    Exact topic phrases and combinations of related terms are
    weighted more strongly than generic words such as 'model'.
    """
    title = clean_text(item.get("filename", "")).lower()
    content = clean_text(item.get("text", "")).lower()
    combined = f"{title} {content}"

    title_phrase_matches = sum(
        1 for phrase in CORE_PHRASES if phrase in title
    )
    content_phrase_matches = sum(
        1 for phrase in CORE_PHRASES if phrase in content
    )

    title_tokens = set(tokenize(title))
    content_tokens = set(tokenize(content))

    title_review = len(title_tokens & REVIEW_TERMS)
    content_review = len(content_tokens & REVIEW_TERMS)

    title_ai = len(title_tokens & AI_TERMS)
    content_ai = len(content_tokens & AI_TERMS)

    content_methods = len(content_tokens & METHOD_TERMS)

    score = (
        title_phrase_matches * 5.0
        + content_phrase_matches * 2.0
        + title_review * 1.5
        + min(content_review, 8) * 0.35
        + title_ai * 0.8
        + min(content_ai, 8) * 0.2
        + min(content_methods, 8) * 0.15
    )

    # A paper-review topic combined with AI or evaluation concepts
    # is stronger evidence than either category alone.
    has_review_topic = bool(
        title_review >= 1 or content_review >= 2
    )
    has_ai_topic = bool(title_ai >= 1 or content_ai >= 2)
    has_review_ai_pair = has_review_topic and has_ai_topic

    if has_review_ai_pair:
        score += 2.0

    # Penalize generic research papers that lack review-related terms.
    if not has_review_topic:
        score *= 0.25

    # Bounded score from zero to one.
    return float(score / (score + 5.0))


def is_relevant_arxiv_paper(item):
    """
    Reject clearly off-topic arXiv papers before final ranking.

    Keep papers with explicit review-topic matches or a meaningful
    combination of review, AI, and evidence/evaluation concepts.
    """
    title = clean_text(item.get("filename", "")).lower()
    content = clean_text(item.get("text", "")).lower()

    title_tokens = set(tokenize(title))
    content_tokens = set(tokenize(content))

    explicit_title_match = any(
        phrase in title for phrase in CORE_PHRASES
    )

    title_review = len(title_tokens & REVIEW_TERMS)
    content_review = len(content_tokens & REVIEW_TERMS)
    title_ai = len(title_tokens & AI_TERMS)
    content_ai = len(content_tokens & AI_TERMS)
    content_methods = len(content_tokens & METHOD_TERMS)

    # Explicitly on-topic titles can pass directly.
    if explicit_title_match:
        return True

    # Require a combination of topical signals in the paper itself.
    review_signal = (
        title_review >= 1 or content_review >= 3
    )
    ai_signal = (
        title_ai >= 1 or content_ai >= 3
    )
    method_signal = content_methods >= 2

    if review_signal and ai_signal:
        return True

    if title_review >= 1 and method_signal and content_ai >= 1:
        return True

    return False


# ============================================================
# REFERENCE PDF LOADING
# ============================================================

@lru_cache(maxsize=1)
def _load_reference_chunks_cached():
    """Load reference PDFs and cache their text chunks."""
    chunks = []

    if not USE_REFERENCE_PDFS:
        return tuple()

    if not REFERENCE_FOLDER.exists():
        print(f"Reference folder not found: {REFERENCE_FOLDER}")
        return tuple()

    pdf_paths = sorted(
        path for path in REFERENCE_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_paths:
        print("No reference PDFs found.")
        return tuple()

    print(f"Loading {len(pdf_paths)} reference PDF(s)...")

    for path in pdf_paths:
        print(f"Processing reference PDF: {path.name}")

        try:
            text = clean_text(extract_text(str(path)))

            if not text:
                print(f"No text extracted from {path.name}.")
                continue

            # Remove the bibliography section when a heading is found.
            matches = list(
                re.finditer(
                    r"(?im)^\s*(?:references|bibliography)\s*$",
                    text,
                )
            )

            if matches:
                text = text[:matches[-1].start()].strip()

            for chunk_id, chunk in enumerate(
                split_text(text), start=1
            ):
                if looks_like_bibliography(chunk):
                    continue

                item = {
                    "filename": path.name,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source_type": "reference_pdf",
                    "source": str(path),
                }
                item["domain_relevance"] = (
                    calculate_domain_relevance(item)
                )
                chunks.append(item)

        except Exception as exc:
            print(f"Error processing {path.name}: {exc}")

    print(f"Reference PDF chunks cached: {len(chunks)}")
    return tuple(chunks)


def load_reference_chunks():
    """Return independent copies of reference chunks."""
    return [
        item.copy() for item in _load_reference_chunks_cached()
    ]


# ============================================================
# REFERENCE PDF TF-IDF INDEX
# ============================================================

@lru_cache(maxsize=1)
def _build_reference_index():
    """Build TF-IDF vectors for reference PDF chunks."""
    chunks = _load_reference_chunks_cached()

    if not chunks:
        return None

    documents = [item["text"] for item in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=np.float32,
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError as exc:
        print(f"Could not build reference index: {exc}")
        return None

    print("Reference TF-IDF index built:", matrix.shape)
    return chunks, vectorizer, matrix


def retrieve_reference_chunks(query, top_k=5):
    """Retrieve the most similar reference PDF chunks."""
    query = prepare_query(query)

    if not query or top_k <= 0:
        return []

    index = _build_reference_index()

    if index is None:
        return []

    chunks, vectorizer, matrix = index
    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        return []

    similarities = cosine_similarity(
        query_vector, matrix
    ).ravel()

    ranked = np.argsort(similarities)[::-1]
    results = []

    for position in ranked:
        similarity = float(similarities[position])

        if similarity < MIN_SIMILARITY:
            break

        item = chunks[position].copy()
        item["similarity"] = similarity
        item["domain_relevance"] = calculate_domain_relevance(item)
        results.append(item)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# ARXIV INDEX LOADING
# ============================================================

@lru_cache(maxsize=1)
def _load_arxiv_index_cached():
    """Load the saved arXiv TF-IDF index."""
    if not USE_ARXIV_DATASET:
        return None

    vectorizer_path = ARXIV_INDEX_FOLDER / "tfidf_vectorizer.pkl"
    matrix_path = ARXIV_INDEX_FOLDER / "tfidf_matrix.pkl"
    documents_path = ARXIV_INDEX_FOLDER / "documents.jsonl"

    required = [
        vectorizer_path,
        matrix_path,
        documents_path,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("arXiv index files missing. Run core/build_dataset_index.py.")
        for path in missing:
            print(f"Missing: {path}")
        return None

    print("Loading saved arXiv index...")

    # Only load pickle files created by trusted project code.
    with vectorizer_path.open("rb") as file:
        vectorizer = pickle.load(file)

    with matrix_path.open("rb") as file:
        matrix = pickle.load(file)

    documents = []

    with documents_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    if matrix.shape[0] != len(documents):
        raise ValueError(
            "arXiv index mismatch: matrix rows differ from document count."
        )

    print(f"arXiv index cached: {len(documents)} papers")
    return vectorizer, matrix, tuple(documents)


def load_arxiv_index():
    """Backward-compatible arXiv index loader."""
    return _load_arxiv_index_cached()


# ============================================================
# ARXIV PAPER RETRIEVAL
# ============================================================

def retrieve_arxiv_chunks(query, top_k=5):
    """Retrieve arXiv papers and remove clearly off-topic results."""
    query = prepare_query(query)

    if not query or top_k <= 0:
        return []

    loaded = load_arxiv_index()

    if loaded is None:
        return []

    vectorizer, matrix, documents = loaded
    query_vector = vectorizer.transform([query])

    if query_vector.nnz == 0:
        return []

    similarities = cosine_similarity(
        query_vector, matrix
    ).ravel()

    ranked = np.argsort(similarities)[::-1]
    results = []

    # Scan beyond top_k because off-topic papers are filtered out.
    scan_limit = min(len(ranked), max(top_k * 12, 60))

    for position in ranked[:scan_limit]:
        similarity = float(similarities[position])

        if similarity < MIN_SIMILARITY:
            continue

        paper = documents[position]

        title = clean_text(
            paper.get("title") or paper.get("filename")
            or "Untitled paper"
        )

        text = clean_text(
            paper.get("text")
            or paper.get("abstract")
            or paper.get("summary")
            or ""
        )

        if not text or looks_like_bibliography(text):
            continue

        item = {
            "filename": title,
            "chunk_id": 1,
            "text": text,
            "source_type": "arxiv_dataset",
            "source": paper.get("source", ""),
            "arxiv_id": paper.get("id", ""),
            "similarity": similarity,
        }

        item["domain_relevance"] = calculate_domain_relevance(item)

        if not is_relevant_arxiv_paper(item):
            continue

        results.append(item)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# RANKING HELPERS
# ============================================================

def _normalize_scores(items, score_key="similarity"):
    """Normalize scores within a result list."""
    if not items:
        return []

    scores = np.array(
        [float(item.get(score_key, 0.0)) for item in items],
        dtype=np.float32,
    )

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum - minimum < 1e-8:
        normalized = np.ones_like(scores)
    else:
        normalized = (scores - minimum) / (maximum - minimum)

    output = []

    for item, score in zip(items, normalized):
        copied = item.copy()
        copied[f"normalized_{score_key}"] = float(score)
        output.append(copied)

    return output


def _make_result_key(item):
    """Create a stable key to prevent duplicate results."""
    source_type = item.get("source_type", "unknown")

    if source_type == "reference_pdf":
        return (
            source_type,
            item.get("source"),
            item.get("chunk_id"),
        )

    return (
        source_type,
        item.get("arxiv_id")
        or item.get("source")
        or item.get("filename"),
        item.get("chunk_id"),
    )


# ============================================================
# DOMAIN-AWARE RETRIEVAL AND RANKING
# ============================================================

def retrieve_chunks(query, top_k=5):
    """
    Retrieve and rank evidence from reference PDFs and arXiv.

    Relevant reference PDFs receive priority, while arXiv papers
    must pass a topical relevance filter before ranking.
    """
    query = prepare_query(query)

    if not query or top_k <= 0:
        return []

    candidate_k = max(
        top_k * CANDIDATE_MULTIPLIER,
        MIN_CANDIDATES,
    )

    pdf_results = retrieve_reference_chunks(query, candidate_k)
    arxiv_results = retrieve_arxiv_chunks(query, candidate_k)

    pdf_results = _normalize_scores(pdf_results)
    arxiv_results = _normalize_scores(arxiv_results)

    fused = {}

    for source_type, source_results in (
        ("reference_pdf", pdf_results),
        ("arxiv_dataset", arxiv_results),
    ):
        for rank, item in enumerate(source_results, start=1):
            key = _make_result_key(item)

            if key not in fused:
                fused[key] = item.copy()
                fused[key]["rrf_score"] = 0.0
                fused[key]["source_rank"] = rank

            # Give directly supplied reference papers more influence.
            source_weight = 1.25 if source_type == "reference_pdf" else 1.0
            fused[key]["rrf_score"] += (
                source_weight / (60 + rank)
            )
            fused[key]["source_rank"] = min(
                fused[key]["source_rank"], rank
            )

    results = list(fused.values())

    if not results:
        return []

    rrf_values = np.array(
        [item.get("rrf_score", 0.0) for item in results],
        dtype=np.float32,
    )

    rrf_min = float(rrf_values.min())
    rrf_max = float(rrf_values.max())

    for item in results:
        similarity_score = float(
            item.get("normalized_similarity", 0.0)
        )
        domain_score = float(
            item.get("domain_relevance", 0.0)
        )
        rrf_score = float(item.get("rrf_score", 0.0))

        if rrf_max - rrf_min < 1e-8:
            normalized_rrf = 1.0
        else:
            normalized_rrf = (
                (rrf_score - rrf_min) / (rrf_max - rrf_min)
            )

        source_type = item.get("source_type")

        # Domain relevance is a major ranking signal.
        relevance_score = (
            0.40 * similarity_score
            + 0.15 * normalized_rrf
            + 0.45 * domain_score
        )

        # Small transparent preference for supplied reference sources.
        if source_type == "reference_pdf":
            relevance_score += 0.05

        item["normalized_rrf"] = float(normalized_rrf)
        item["relevance_score"] = float(relevance_score)

    results.sort(
        key=lambda item: (
            item.get("relevance_score", 0.0),
            item.get("similarity", 0.0),
        ),
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# CACHE RESET
# ============================================================

def clear_retrieval_caches():
    """Clear cached indexes after updating PDFs or the dataset."""
    _build_reference_index.cache_clear()
    _load_reference_chunks_cached.cache_clear()
    _load_arxiv_index_cached.cache_clear()

    print("RAG retrieval caches cleared.")
