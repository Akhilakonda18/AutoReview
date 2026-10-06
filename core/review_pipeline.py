from core.pdf_parser import extract_text
from core.section_identifier import identify_sections
from core.rag import retrieve_chunks
from core.score_fusion import ScoreFusion

from agents.agent_manager import AgentManager
from agents.reviewer_agent import ReviewerAgent


# ---------------------------------------------------------
# Research paper to review
# ---------------------------------------------------------

PDF_PATH = "data/sample papers/paper.pdf"


def run_full_review():

    print("=" * 70)
    print("AUTOREVIEW - COMPLETE RESEARCH PAPER REVIEW")
    print("=" * 70)

    # =====================================================
    # 1. Extract paper text
    # =====================================================

    print("\n[1] Reading research paper...")

    paper_text = extract_text(PDF_PATH)

    print(
        "Characters extracted:",
        len(paper_text)
    )

    # =====================================================
    # 2. Identify paper sections
    # =====================================================

    print("\n[2] Identifying paper sections...")

    sections = identify_sections(paper_text)

    print("Sections detected:")

    for section in sections.keys():
        print(" -", section)

    # =====================================================
    # 3. Retrieve reference evidence using RAG
    # =====================================================

    print("\n[3] Retrieving reference evidence...")

    query = """
    Research paper structure, literature review,
    research methodology, experiments, results,
    citations, novelty, scientific writing,
    research contributions, limitations,
    and peer review.
    """

    retrieved = retrieve_chunks(
        query,
        top_k=3
    )

    evidence = ""

    for result in retrieved:

        evidence += f"""
Reference Paper: {result['filename']}
Similarity: {result['similarity']}

Evidence:
{result['text'][:1000]}

----------------------------------------
"""

    print(
        "Reference evidence collected:",
        len(retrieved),
        "chunks"
    )

    # =====================================================
    # 4. Run all specialist agents
    # =====================================================

    print("\n[4] Running all specialist agents...")

    manager = AgentManager()

    results = manager.run_all(
        paper_text,
        evidence
    )

    # =====================================================
    # 5. Score Fusion
    # =====================================================

    print("\n")
    print("=" * 70)
    print("SCORE FUSION")
    print("=" * 70)

    fusion = ScoreFusion()

    overall_score = fusion.calculate_overall_score(
        results
    )

    print("\nIndividual Agent Scores:")

    for agent_name, score in fusion.scores.items():

        print(
            f"{agent_name}: {score}/10"
        )

    print(
        "\nOverall Score:",
        overall_score,
        "/10"
    )

    # =====================================================
    # 6. Final Reviewer Agent
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL REVIEWER AGENT")
    print("=" * 70)

    reviewer = ReviewerAgent()

    # Convert agent results into readable text
    agent_results_text = ""

    for agent_name, review in results.items():

        agent_results_text += f"""

==================================================
{agent_name}
==================================================

{review}

"""

    final_result = reviewer.review(
        paper_text,
        agent_results_text,
        fusion.scores,
        overall_score
    )

    # =====================================================
    # 7. Display final consolidated review
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL CONSOLIDATED REVIEW")
    print("=" * 70)

    print(
        final_result["review"]
    )

    # =====================================================
    # 8. Return complete results
    # =====================================================

    return {
        "paper_text": paper_text,
        "sections": sections,
        "retrieved_evidence": evidence,
        "agent_results": results,
        "agent_scores": fusion.scores,
        "overall_score": overall_score,
        "final_review": final_result["review"]
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    run_full_review()