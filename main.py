
from core.pdf_parser import extract_text
from core.section_identifier import identify_sections
from core.rag import retrieve_chunks
from core.score_fusion import ScoreFusion
from core.report_generator import generate_report
from core.evidence_verifier import verify_review
from core.review_pipeline import AGENT_QUERIES, build_agent_query, format_evidence

from pathlib import Path
from plagiarism_checker import run_plagiarism_check

from agents.agent_manager import AgentManager
from agents.final_reviewer import FinalReviewer


# =========================================================
# CONFIGURATION
# =========================================================

PDF_PATH = "data/sample papers/paper.pdf"
REFERENCE_DIRS = [Path("data/reference papers"), Path("data/reference_papers")]
RUN_PLAGIARISM_CHECK = True


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_list(value):
    """Convert a value into a list of readable strings."""

    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    value = str(value).strip()
    return [value] if value else []


def build_verification_summary(verification_results):
    """Create a readable verification summary for the final report."""

    lines = [
        "",
        "",
        "=" * 60,
        "EVIDENCE VERIFICATION AND SELF-CORRECTION SUMMARY",
        "=" * 60,
    ]

    for agent_name, verification in verification_results.items():
        lines.append("")
        lines.append(f"Agent: {agent_name}")
        lines.append(
            f"Final verification status: "
            f"{verification.get('status', 'UNVERIFIED')}"
        )
        lines.append(
            f"Correction attempts: {verification.get('attempts', 0)}"
        )

        feedback = verification.get("feedback", "")
        if feedback:
            lines.append(f"Verifier feedback: {feedback}")

        unsupported = normalize_list(
            verification.get("unsupported_claims", [])
        )
        if unsupported:
            lines.append("Claims requiring further checking:")
            for claim in unsupported:
                lines.append(f"- {claim}")

        missing = normalize_list(
            verification.get("missing_evidence", [])
        )
        if missing:
            lines.append("Missing evidence:")
            for item in missing:
                lines.append(f"- {item}")

    lines.extend([
        "",
        "Note: LLM-based verification is an aid, not a guarantee "
        "of factual correctness. Unverified claims require human review.",
        "=" * 60,
    ])

    return "\n".join(lines)


def build_plagiarism_summary(plagiarism_result):
    """Format plagiarism-check results for the consolidated AutoReview report."""

    if not isinstance(plagiarism_result, dict):
        return (
            "PLAGIARISM SIMILARITY CHECK\n"
            "Status: NOT COMPLETED\n"
            "The plagiarism checker did not return a result. Please review its error."
        )

    summary = plagiarism_result.get("summary", {})
    lines = [
        "",
        "=" * 60,
        "PLAGIARISM SIMILARITY CHECK",
        "=" * 60,
        f"Status: {summary.get('status', 'UNKNOWN')}",
        f"Main-text words checked: {summary.get('submitted_main_text_word_count', 'N/A')}",
        f"Exact-match records: {summary.get('exact_match_records', 0)}",
        f"TF-IDF match records: {summary.get('tfidf_match_records', 0)}",
        f"Semantic match records: {summary.get('semantic_match_records', 0)}",
        f"Possible duplicate bibliography entries: {summary.get('bibliographic_possible_duplicate_entries', 0)}",
        f"Estimated flagged-word coverage: {summary.get('estimated_flagged_word_coverage_percent', 0)}%",
    ]

    checked = plagiarism_result.get("reference_pdfs_checked", [])
    if checked:
        lines.append(f"Reference PDFs checked: {len(checked)}")
        for reference in checked:
            lines.append(f"- {reference.get('name', 'Unknown reference')}")

    if plagiarism_result.get("txt_report"):
        lines.append(f"Detailed text report: {plagiarism_result['txt_report']}")
    if plagiarism_result.get("json_report"):
        lines.append(f"Detailed JSON report: {plagiarism_result['json_report']}")

    lines.append(
        summary.get(
            "note",
            "Similarity results require human review and do not prove or disprove plagiarism.",
        )
    )
    lines.append("=" * 60)
    return "\n".join(lines)


# =========================================================
# MAIN REVIEW FUNCTION
# =========================================================

def run_full_review():

    print("=" * 70)
    print("AUTOREVIEW - COMPLETE RESEARCH PAPER REVIEW")
    print("=" * 70)

    # =====================================================
    # 1. EXTRACT RESEARCH PAPER
    # =====================================================

    print("\n[1] Reading research paper...")

    try:
        paper_text = extract_text(PDF_PATH)

        if not paper_text or not paper_text.strip():
            raise ValueError("No text was extracted from the PDF.")

    except Exception as error:
        print("\nERROR reading research paper:")
        print(error)
        return None

    print("Characters extracted:", len(paper_text))
    print("Words extracted:", len(paper_text.split()))


    # =====================================================
    # 1A. PLAGIARISM SIMILARITY CHECK
    # =====================================================

    plagiarism_result = None
    plagiarism_summary = ""

    if RUN_PLAGIARISM_CHECK:
        print("\n[1A] Running plagiarism similarity checker...")
        try:
            existing_reference_dirs = [str(path) for path in REFERENCE_DIRS if path.is_dir()]
            if not existing_reference_dirs:
                existing_reference_dirs = None

            plagiarism_result = run_plagiarism_check(
                PDF_PATH,
                reference_dirs=existing_reference_dirs,
                offline=True,
            )
            plagiarism_summary = build_plagiarism_summary(plagiarism_result)
            print(plagiarism_summary)
        except Exception as error:
            print("\nPLAGIARISM CHECK ERROR:")
            print(error)
            plagiarism_summary = (
                "PLAGIARISM SIMILARITY CHECK\n"
                "Status: NOT COMPLETED\n"
                f"Error: {error}\n"
                "The rest of AutoReview will continue."
            )
    else:
        plagiarism_summary = (
            "PLAGIARISM SIMILARITY CHECK\n"
            "Status: SKIPPED (RUN_PLAGIARISM_CHECK is False)"
        )

    # =====================================================
    # 2. IDENTIFY PAPER SECTIONS
    # =====================================================

    print("\n[2] Identifying paper sections...")

    try:
        sections = identify_sections(paper_text)

        if not isinstance(sections, dict):
            sections = {}

    except Exception as error:
        print("\nSECTION IDENTIFICATION ERROR:")
        print(error)
        sections = {}

    print("Sections detected:")

    for section in sections:
        print(" -", section)

    # =====================================================
    # 3. RETRIEVE REFERENCE EVIDENCE
    # =====================================================

    print("\n[3] Retrieving reference evidence...")

    evidence_by_agent = {}

    try:
        for agent_name in AGENT_QUERIES:
            query = build_agent_query(agent_name, paper_text, sections)
            try:
                retrieved = retrieve_chunks(query, top_k=5) or []
            except Exception as error:
                print(f"Warning: RAG retrieval failed for {agent_name}: {error}")
                retrieved = []

            evidence_by_agent[agent_name] = format_evidence(retrieved)
            print(
                f"Retrieved {len(retrieved)} evidence chunks "
                f"for {agent_name}"
            )

    except Exception as error:
        print("\nRAG ERROR:")
        print(error)
        evidence_by_agent = {
            agent_name: "No relevant reference evidence was retrieved."
            for agent_name in AGENT_QUERIES
        }

    print("Agent-specific reference retrieval completed.")

    # =====================================================
    # 4. RUN SPECIALIST AGENTS
    # =====================================================

    print("\n[4] Running all specialist agents...")

    manager = AgentManager()

    results = manager.run_all(paper_text, evidence_by_agent)

    if not isinstance(results, dict) or not results:
        print("\nERROR:")
        print("No specialist agents produced results.")
        return None

    # =====================================================
    # 4A. EVIDENCE VERIFICATION AND SELF-CORRECTION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("EVIDENCE VERIFICATION AND SELF-CORRECTION")
    print("=" * 70)

    verification_results = {}

    for agent in manager.agents:

        agent_name = agent.name

        if agent_name not in results:
            verification_results[agent_name] = {
                "status": "NOT_RUN",
                "attempts": 0,
                "feedback": "No specialist review was generated.",
                "unsupported_claims": [],
                "missing_evidence": [],
            }
            continue

        print(f"\nVerifying: {agent_name}")

        current_review = results[agent_name]
        attempts = 0

        try:
            agent_evidence = evidence_by_agent.get(agent_name, "No relevant reference evidence was retrieved.")
            verification = verify_review(current_review, agent_evidence, paper_text=paper_text)

            if not isinstance(verification, dict):
                verification = {
                    "status": "UNVERIFIED",
                    "feedback": "Verifier returned an invalid result.",
                    "unsupported_claims": [],
                    "missing_evidence": [],
                }

            status = str(
                verification.get("status", "UNVERIFIED")
            ).strip().upper()

            print("Verification status:", status)

            # Attempt correction only when revision is requested.
            if status == "NEEDS_REVISION":

                attempts = 1

                unsupported_claims = normalize_list(
                    verification.get("unsupported_claims", [])
                )

                missing_evidence = normalize_list(
                    verification.get("missing_evidence", [])
                )

                feedback = str(
                    verification.get("feedback", "")
                ).strip()

                correction_instructions = f"""
SELF-CORRECTION INSTRUCTIONS
These instructions are feedback, not research evidence.

Verification feedback:
{feedback or "Review the unsupported claims and evidence gaps."}

Claims to re-check:
{chr(10).join("- " + item for item in unsupported_claims)
 or "- Recheck important factual claims."}

Evidence gaps to address:
{chr(10).join("- " + item for item in missing_evidence)
 or "- Identify claims that lack adequate evidence."}

Requirements:
1. Revise only where necessary.
2. Do not invent facts, citations, quotations, or experimental results.
3. Do not claim that a reference supports a statement unless the supplied
   evidence actually supports it.
4. Clearly distinguish statements about the submitted paper from claims
   about external research.
5. If evidence is insufficient, acknowledge the limitation rather than
   pretending the claim has been verified.
6. Preserve the specialist review's original purpose and structure.
"""

                print(f"Attempting correction for {agent_name}...")

                corrected_name, corrected_review = (
                    manager.run_single_agent(
                        agent,
                        paper_text,
                        agent_evidence + "\n\n" + correction_instructions,
                    )
                )

                if corrected_review and str(corrected_review).strip():

                    # Replace the original only when a correction exists.
                    results[agent_name] = corrected_review

                    print(f"Rechecking corrected review: {agent_name}")

                    verification = verify_review(
                        corrected_review,
                        agent_evidence,
                        paper_text=paper_text,
                    )

                    if not isinstance(verification, dict):
                        verification = {
                            "status": "UNVERIFIED",
                            "feedback": (
                                "Verifier returned an invalid result "
                                "after correction."
                            ),
                            "unsupported_claims": [],
                            "missing_evidence": [],
                        }

                    final_status = str(
                        verification.get("status", "UNVERIFIED")
                    ).strip().upper()

                    print("Verification after correction:", final_status)

                else:
                    verification = {
                        "status": "CORRECTION_FAILED",
                        "feedback": (
                            "The specialist did not produce a corrected review."
                        ),
                        "unsupported_claims": unsupported_claims,
                        "missing_evidence": missing_evidence,
                    }

            else:
                # SUPPORTED and UNVERIFIED are retained without correction.
                verification["status"] = status

        except Exception as error:

            print(f"Verification/correction error for {agent_name}: {error}")

            verification = {
                "status": "UNVERIFIED",
                "feedback": str(error),
                "unsupported_claims": [],
                "missing_evidence": [],
            }

        verification["attempts"] = attempts
        verification_results[agent_name] = verification

    # =====================================================
    # 5. SCORE FUSION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("SCORE FUSION")
    print("=" * 70)

    fusion = ScoreFusion()
    overall_score = fusion.calculate_overall_score(results)

    print("\nIndividual Agent Scores:")

    for agent_name, score in fusion.scores.items():
        print(f"{agent_name}: {score}/10")

    print(
        "\nSuccessful agents:",
        len(fusion.scores),
        "/",
        len(manager.agents),
    )

    print("\nOverall Score:", overall_score, "/10")

    # =====================================================
    # 6. FINAL REVIEWER
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
            overall_score,
        )

    except Exception as error:
        print("\nERROR while running Final Reviewer:")
        print(error)

        final_review = (
            "Final Reviewer Agent failed.\n\n"
            f"Error: {error}"
        )

    # Add the verification summary so it is included in the report.
    verification_summary = build_verification_summary(
        verification_results
    )

    final_review_with_verification = (
        str(final_review) + "\n\n" + verification_summary
        + "\n\n" + plagiarism_summary
    )

    # =====================================================
    # 7. DISPLAY FINAL REVIEW
    # =====================================================

    print("\n")
    print("=" * 70)
    print("FINAL CONSOLIDATED REVIEW")
    print("=" * 70)

    print(final_review_with_verification)

    # =====================================================
    # 8. GENERATE FINAL REPORT
    # =====================================================

    print("\n")
    print("=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    report_path = None

    try:
        report_path = generate_report(
            PDF_PATH,
            results,
            overall_score,
            final_review_with_verification,
        )

        print("\nReport generated successfully!")
        print("Saved at:", report_path)

    except Exception as error:
        print("\nREPORT GENERATION ERROR:")
        print(error)

    # =====================================================
    # 9. DISPLAY VERIFICATION SUMMARY
    # =====================================================

    print("\n")
    print("=" * 70)
    print("SELF-CORRECTION SUMMARY")
    print("=" * 70)

    for agent_name, verification in verification_results.items():
        print(f"\n{agent_name}")
        print("Status:", verification.get("status", "UNVERIFIED"))
        print("Correction attempts:", verification.get("attempts", 0))

        feedback = verification.get("feedback", "")
        if feedback:
            print("Feedback:", feedback)

    # =====================================================
    # 10. COMPLETED
    # =====================================================

    print("\n")
    print("=" * 70)
    print("AUTOREVIEW COMPLETED")
    print("=" * 70)

    return {
        "paper_text": paper_text,
        "sections": sections,
        "retrieved_evidence": evidence_by_agent,
        "agent_results": results,
        "agent_scores": fusion.scores,
        "overall_score": overall_score,
        "verification_results": verification_results,
        "plagiarism_result": plagiarism_result,
        "plagiarism_summary": plagiarism_summary,
        "final_review": final_review_with_verification,
        "report_path": report_path,
    }


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_full_review()
