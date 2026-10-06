from core.pdf_parser import extract_text
from core.section_identifier import identify_sections
from core.rag import retrieve_chunks
from core.score_fusion import ScoreFusion
from core.report_generator import generate_report

from agents.agent_manager import AgentManager
from agents.final_reviewer import FinalReviewer


# =========================================================
# CONFIGURATION
# =========================================================

PDF_PATH = "data/sample papers/paper.pdf"


# =========================================================
# MAIN REVIEW FUNCTION
# =========================================================

def run_full_review():

    print("=" * 70)
    print("AUTOREVIEW - COMPLETE RESEARCH PAPER REVIEW")
    print("=" * 70)

    # =====================================================
    # 1. Extract Research Paper
    # =====================================================

    print("\n[1] Reading research paper...")

    try:

        paper_text = extract_text(PDF_PATH)

    except Exception as e:

        print("\nERROR reading research paper:")
        print(e)

        return None

    print(
        "Characters extracted:",
        len(paper_text)
    )

    print(
        "Words extracted:",
        len(paper_text.split())
    )

    # =====================================================
    # 2. Identify Sections
    # =====================================================

    print("\n[2] Identifying paper sections...")

    try:

        sections = identify_sections(paper_text)

    except Exception as e:

        print("\nSECTION IDENTIFICATION ERROR:")
        print(e)

        sections = {}

    print("Sections detected:")

    for section in sections.keys():

        print(" -", section)

    # =====================================================
    # 3. Retrieve Reference Evidence
    # =====================================================

    print("\n[3] Retrieving reference evidence...")

    query = """
    Research paper structure,
    literature review,
    research methodology,
    experiments,
    results,
    citations,
    novelty,
    scientific writing,
    research contributions,
    limitations,
    and peer review.
    """

    try:

        retrieved = retrieve_chunks(
            query,
            top_k=3
        )

    except Exception as e:

        print("\nRAG ERROR:")
        print(e)

        retrieved = []

    evidence = ""

    for result in retrieved:

        evidence += f"""

Reference Paper:
{result['filename']}

Similarity:
{result['similarity']}

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
    # 4. Run Specialist Agents
    # =====================================================

    print("\n[4] Running all specialist agents...")

    manager = AgentManager()

    results = manager.run_all(
        paper_text,
        evidence
    )

    # =====================================================
    # Check Specialist Results
    # =====================================================

    if not results:

        print("\nERROR:")
        print("No specialist agents produced results.")

        return None

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
            f"{agent_name}: "
            f"{score}/10"
        )

    print(
        "\nSuccessful agents:",
        len(fusion.scores),
        "/ 7"
    )

    print(
        "\nOverall Score:",
        overall_score,
        "/10"
    )

    # =====================================================
    # 6. Final Reviewer
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL REVIEWER AGENT")
    print("=" * 70)

    try:

        reviewer = FinalReviewer()

        final_review = reviewer.review(
            paper_text,
            results,
            overall_score
        )

    except Exception as e:

        print("\nERROR while running Final Reviewer:")
        print(e)

        final_review = (
            "Final Reviewer Agent failed.\n\n"
            f"Error: {e}"
        )

    # =====================================================
    # 7. Display Final Review
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL CONSOLIDATED REVIEW")
    print("=" * 70)

    print(final_review)

    # =====================================================
    # 8. Generate Final Report
    # =====================================================

    print("\n")
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    try:

        paper_name = PDF_PATH

        report_path = generate_report(
            paper_name,
            results,
            overall_score,
            final_review
        )

        print(
            "\nReport generated successfully!"
        )

        print(
            "Saved at:",
            report_path
        )

    except Exception as e:

        print("\nREPORT GENERATION ERROR:")
        print(e)

        report_path = None

    # =====================================================
    # 9. Completed
    # =====================================================

    print("\n")
    print("=" * 70)
    print("AUTOREVIEW COMPLETED")
    print("=" * 70)

    return {

        "paper_text":
            paper_text,

        "sections":
            sections,

        "retrieved_evidence":
            evidence,

        "agent_results":
            results,

        "agent_scores":
            fusion.scores,

        "overall_score":
            overall_score,

        "final_review":
            final_review,

        "report_path":
            report_path
    }


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    run_full_review()