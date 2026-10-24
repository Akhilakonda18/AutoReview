
import re
from pathlib import Path

from core.pdf_parser import extract_text
from core.section_identifier import identify_sections
from core.rag import retrieve_chunks
from agents.agent_manager import AgentManager
from agents.reviewer_agent import ReviewerAgent

# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/sample papers/paper.pdf"
TOP_K_PER_AGENT = 5
MAX_SECTION_CHARS = 5000
MAX_EVIDENCE_CHARS = 1800
DEBUG_EVIDENCE_CHARS = 500

# ============================================================
# SPECIALIST AGENT OBJECTIVES
# ============================================================

AGENT_QUERIES = {
    "Structure Agent": {
        "objective": (
            "Evaluate research paper organization, logical flow, "
            "section completeness, and structural consistency."
        ),
        "sections": [
            "abstract", "introduction", "methodology",
            "results", "conclusion",
        ],
    },
    "Novelty Agent": {
        "objective": (
            "Evaluate research originality, research contributions, "
            "novel ideas, and differences from existing approaches."
        ),
        "sections": [
            "abstract", "introduction", "related work", "conclusion",
        ],
    },
    "Methodology Agent": {
        "objective": (
            "Evaluate the research methodology, proposed approach, "
            "algorithm design, implementation details, and reproducibility."
        ),
        "sections": [
            "methodology", "proposed method",
            "system architecture", "experimental setup",
        ],
    },
    "Literature Agent": {
        "objective": (
            "Evaluate the literature review, related work, coverage of "
            "previous research, comparison with existing methods, "
            "and identification of research gaps."
        ),
        "sections": [
            "related work", "literature review",
            "introduction", "references",
        ],
    },
    "Experiment Agent": {
        "objective": (
            "Evaluate experimental design, datasets, baselines, evaluation "
            "metrics, results, and the strength of supporting evidence."
        ),
        "sections": [
            "experimental setup", "experiments",
            "results", "evaluation", "discussion",
        ],
    },
    "Writing Agent": {
        "objective": (
            "Evaluate academic writing quality, clarity, grammar, "
            "readability, terminology, and consistency."
        ),
        "sections": [
            "abstract", "introduction", "methodology",
            "results", "conclusion",
        ],
    },
    "Citation Agent": {
        "objective": (
            "Evaluate citation quality, reference completeness, "
            "citation consistency, and whether research claims "
            "are appropriately supported."
        ),
        "sections": [
            "related work", "literature review",
            "methodology", "references",
        ],
    },
}

# ============================================================
# PAPER SECTION HELPERS
# ============================================================

def normalize_text(text):
    """Normalize whitespace without changing the text's meaning."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def find_section(sections, target_name):
    """Find a section using case-insensitive, partial name matching."""
    if not isinstance(sections, dict):
        return ""

    target = target_name.lower()

    for name, content in sections.items():
        normalized_name = str(name).lower().strip()

        if target == normalized_name or target in normalized_name:
            if isinstance(content, str):
                return content

            if isinstance(content, dict):
                return str(
                    content.get("text", content.get("content", ""))
                )

    return ""


def build_agent_query(agent_name, paper_text, sections):
    """
    Build a focused agent-specific query.
    Do not append the research paper abstract or full paper text.
    """
    retrieval_terms = {
        "Structure Agent": (
            "scientific paper organization section structure logical flow "
            "academic writing completeness"
        ),
        "Novelty Agent": (
            "research novelty originality contributions comparison "
            "existing methods research gaps"
        ),
        "Methodology Agent": (
            "research methodology algorithms system architecture "
            "implementation reproducibility"
        ),
        "Literature Agent": (
            "automated scientific peer review related work literature review "
            "previous research comparison research gaps"
        ),
        "Experiment Agent": (
            "experimental evaluation datasets baselines metrics results "
            "ablation studies statistical significance"
        ),
        "Writing Agent": (
            "academic scientific writing clarity grammar readability "
            "terminology consistency"
        ),
        "Citation Agent": (
            "academic citations references bibliography citation completeness "
            "evidence supported claims"
        ),
    }

    shared_terms = (
        "automated scientific peer review manuscript assessment "
        "large language models LLM reviewer feedback RAG "
        "evidence grounded critique"
    )

    config = AGENT_QUERIES.get(agent_name, {})
    objective = config.get(
        "objective",
        "Evaluate the quality and technical correctness of a scientific paper.",
    )
    agent_terms = retrieval_terms.get(
        agent_name,
        "scientific research paper evaluation",
    )

    query = f"{shared_terms} {agent_terms} {objective}"
    return normalize_text(query)[:1200]


# ============================================================
# EVIDENCE FORMATTING
# ============================================================

def get_result_value(result, possible_keys, default=""):
    """Read a field from a dictionary or an object."""
    if isinstance(result, dict):
        for key in possible_keys:
            if key in result and result[key] is not None:
                return result[key]

    for key in possible_keys:
        if hasattr(result, key):
            value = getattr(result, key)
            if value is not None:
                return value

    return default


def format_evidence(results):
    """Format retrieved evidence for the specialist agents."""
    if not results:
        return "No relevant reference evidence was retrieved."

    formatted = []

    for index, result in enumerate(results, start=1):
        if isinstance(result, str):
            formatted.append(
                f"[Evidence {index}]\n{result[:MAX_EVIDENCE_CHARS]}"
            )
            continue

        source_type = get_result_value(
            result, ["source_type", "source", "type"], "Unknown"
        )
        filename = get_result_value(
            result, ["filename", "file_name", "title", "document"], "Unknown"
        )
        similarity = get_result_value(
            result, ["similarity", "similarity_score", "tfidf_score"], None
        )
        rrf_score = get_result_value(
            result, ["rrf_score", "rrf", "score"], None
        )
        content = get_result_value(
            result, ["text", "content", "chunk", "page_content"], ""
        )

        if not content:
            content = str(result)

        item = [
            f"[Evidence {index}]",
            f"Source type: {source_type}",
            f"Source: {filename}",
        ]

        if similarity is not None:
            item.append(f"Similarity: {similarity}")

        if rrf_score is not None:
            item.append(f"RRF score: {rrf_score}")

        item.append(
            "Evidence text:\n"
            + normalize_text(content)[:MAX_EVIDENCE_CHARS]
        )
        formatted.append("\n".join(item))

    return "\n\n".join(formatted)


# ============================================================
# DEBUGGING RETRIEVED EVIDENCE
# ============================================================

def print_retrieved_evidence(agent_name, retrieved):
    """Print retrieval metadata and a short text preview."""
    print(f"\n--- Retrieved evidence: {agent_name} ---")

    if not retrieved:
        print("No evidence returned.")
        return

    for index, result in enumerate(retrieved, start=1):
        print(f"\nEvidence {index}")

        if isinstance(result, str):
            print("Source: Unknown")
            print(
                "Text preview: "
                f"{normalize_text(result)[:DEBUG_EVIDENCE_CHARS]}"
            )
            continue

        source = get_result_value(
            result, ["filename", "file_name", "title", "document"], "Unknown"
        )
        source_type = get_result_value(
            result, ["source_type", "source", "type"], "Unknown"
        )
        similarity = get_result_value(
            result,
            ["similarity", "similarity_score", "tfidf_score"],
            "Not provided",
        )
        rrf_score = get_result_value(
            result, ["rrf_score", "rrf", "score"], "Not provided"
        )
        content = get_result_value(
            result, ["text", "content", "chunk", "page_content"], ""
        )

        print(f"Source: {source}")
        print(f"Source type: {source_type}")
        print(f"Similarity: {similarity}")
        print(f"RRF score: {rrf_score}")
        print(
            "Text preview: "
            f"{normalize_text(content)[:DEBUG_EVIDENCE_CHARS]}"
        )


# ============================================================
# TARGETED RAG RETRIEVAL
# ============================================================

def retrieve_agent_evidence(paper_text, sections):
    """Retrieve evidence separately for each specialist agent."""
    evidence_by_agent = {}

    print("\n" + "=" * 65)
    print("TARGETED RAG EVIDENCE RETRIEVAL")
    print("=" * 65)

    for agent_name in AGENT_QUERIES:
        print(f"\nRetrieving evidence for: {agent_name}")

        query = build_agent_query(agent_name, paper_text, sections)

        try:
            retrieved = retrieve_chunks(
                query,
                top_k=TOP_K_PER_AGENT,
            )

            if retrieved is None:
                retrieved = []

            evidence_by_agent[agent_name] = retrieved

            print(
                f"Retrieved {len(retrieved)} evidence chunks "
                f"for {agent_name}"
            )
            print_retrieved_evidence(agent_name, retrieved)

        except Exception as error:
            print(
                f"Warning: Retrieval failed for {agent_name}: {error}"
            )
            evidence_by_agent[agent_name] = []

    return evidence_by_agent


# ============================================================
# SPECIALIST SCORE EXTRACTION
# ============================================================

def extract_agent_score(agent_result):
    """Extract a score out of 10 from a specialist's review."""
    if isinstance(agent_result, dict):
        review_text = str(
            agent_result.get(
                "review",
                agent_result.get("output", agent_result),
            )
        )
    else:
        review_text = str(agent_result)

    patterns = [
        r"\bOVERALL\s+SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\bFINAL\s+SCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\bSCORE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10\b",
        r"\b(\d+(?:\.\d+)?)\s*/\s*10\b",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            review_text,
            flags=re.IGNORECASE,
        )
        if matches:
            score = float(matches[0])
            if 0 <= score <= 10:
                return score

    return None


def calculate_overall_score(individual_scores):
    """Calculate the arithmetic mean of available specialist scores."""
    valid_scores = [
        float(score)
        for score in individual_scores.values()
        if score is not None and 0 <= float(score) <= 10
    ]

    if not valid_scores:
        raise ValueError(
            "No valid specialist scores were extracted. "
            "Check the specialist agent output format."
        )

    return round(sum(valid_scores) / len(valid_scores), 2)


# ============================================================
# MAIN REVIEW PIPELINE
# ============================================================

def run_full_review(pdf_path=PDF_PATH):
    """
    Execute the complete AutoReview pipeline:
    1. Extract paper text.
    2. Identify paper sections.
    3. Retrieve targeted evidence for each specialist.
    4. Run specialist review agents.
    5. Extract scores and calculate the overall score.
    6. Generate the final consolidated review.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"Research paper not found: {pdf_file.resolve()}\n"
            "Check the PDF_PATH or pass the correct PDF path."
        )

    # STEP 1: EXTRACT PAPER TEXT
    print("\n" + "=" * 65)
    print("STEP 1: PAPER TEXT EXTRACTION")
    print("=" * 65)

    paper_text = extract_text(str(pdf_file))

    if not paper_text or not paper_text.strip():
        raise ValueError(
            "No text was extracted from the PDF. "
            "Check whether the PDF is scanned or image-only."
        )

    print(f"Paper: {pdf_file.name}")
    print(f"Extracted characters: {len(paper_text):,}")

    # STEP 2: IDENTIFY PAPER SECTIONS
    print("\n" + "=" * 65)
    print("STEP 2: SECTION IDENTIFICATION")
    print("=" * 65)

    sections = identify_sections(paper_text)

    if not isinstance(sections, dict):
        print(
            "Warning: identify_sections did not return a dictionary. "
            "Using an empty section mapping."
        )
        sections = {}

    print(f"Sections detected: {len(sections)}")
    for section_name in sections:
        print(f"  - {section_name}")

    # STEP 3: RETRIEVE TARGETED EVIDENCE
    evidence_by_agent = retrieve_agent_evidence(paper_text, sections)

    formatted_evidence = {
        agent_name: format_evidence(results)
        for agent_name, results in evidence_by_agent.items()
    }

    # STEP 4: RUN SPECIALIST AGENTS
    print("\n" + "=" * 65)
    print("STEP 3: SPECIALIST AGENT REVIEWS")
    print("=" * 65)

    manager = AgentManager()
    results = manager.run_all(paper_text, evidence_by_agent)

    if results is None:
        results = {}

    if not isinstance(results, dict):
        raise TypeError(
            "AgentManager.run_all() must return a dictionary "
            "of specialist agent results."
        )

    # STEP 5: EXTRACT SPECIALIST SCORES
    print("\n" + "=" * 65)
    print("STEP 4: SPECIALIST SCORE EXTRACTION")
    print("=" * 65)

    individual_scores = {}

    for agent_name, agent_result in results.items():
        score = extract_agent_score(agent_result)
        individual_scores[agent_name] = score

        if score is None:
            print(f"{agent_name}: score not found")
        else:
            print(f"{agent_name}: {score:.2f}/10")

    # STEP 6: CALCULATE OVERALL SCORE
    print("\n" + "=" * 65)
    print("STEP 5: OVERALL SCORE CALCULATION")
    print("=" * 65)

    overall_score = calculate_overall_score(individual_scores)

    scores = {
        "individual_scores": individual_scores,
        "overall_score": overall_score,
    }

    print(f"Overall score: {overall_score:.2f}/10")

    # STEP 7: FINAL CONSOLIDATED REVIEW
    print("\n" + "=" * 65)
    print("STEP 6: FINAL REVIEWER")
    print("=" * 65)

    reviewer = ReviewerAgent()

    final_review = reviewer.review(
        paper_text=paper_text,
        agent_results=results,
        agent_scores=individual_scores,
        overall_score=overall_score,
    )

    # STEP 8: PRINT FINAL RESULT
    print("\n" + "=" * 65)
    print("FINAL AUTOREVIEW REPORT")
    print("=" * 65)
    print(f"\nCALCULATED OVERALL SCORE: {overall_score:.2f}/10")

    if isinstance(final_review, dict):
        review_text = final_review.get("review", str(final_review))
    else:
        review_text = str(final_review)

    print("\n")
    print(review_text)

    return {
        "paper_path": str(pdf_file),
        "paper_text": paper_text,
        "sections": sections,
        "evidence_by_agent": evidence_by_agent,
        "formatted_evidence": formatted_evidence,
        "agent_results": results,
        "scores": scores,
        "final_review": final_review,
    }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        report = run_full_review()
        print("\nReview pipeline completed.")
    except Exception as error:
        print("\nReview pipeline failed.")
        print(f"Error: {error}")
        raise
