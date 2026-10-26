
from core.llm_service import ask_qwen


class FinalReviewer:

    def __init__(self):
        self.name = "Final Reviewer Agent"

    @staticmethod
    def _review_to_text(review):
        """Convert a specialist result into readable text."""

        if isinstance(review, dict):
            return str(
                review.get("review")
                or review.get("content")
                or review
            )

        return str(review or "")

    @staticmethod
    def _verification_to_text(verification_results):
        """Format verification results for the final reviewer."""

        if not verification_results:
            return (
                "No verification results were supplied. "
                "Treat specialist findings as unverified."
            )

        blocks = []

        for agent_name, result in verification_results.items():

            if not isinstance(result, dict):
                blocks.append(
                    f"{agent_name}: UNVERIFIED "
                    "(invalid result format)"
                )
                continue

            status = str(
                result.get("status", "UNVERIFIED")
            ).strip().upper()

            if status not in {
                "SUPPORTED",
                "NEEDS_REVISION",
                "UNVERIFIED",
            }:
                status = "UNVERIFIED"

            blocks.append(
                f"""
AGENT: {agent_name}

VERIFICATION STATUS:
{status}

UNSUPPORTED CLAIMS:
{result.get("unsupported_claims", [])}

MISSING EVIDENCE:
{result.get("missing_evidence", [])}

VERIFIER FEEDBACK:
{result.get("feedback", "")}
"""
            )

        return "\n\n".join(blocks)

    def build_prompt(
        self,
        paper_text,
        specialist_results,
        overall_score,
        verification_results=None,
    ):

        specialist_blocks = []

        for agent_name, review in (
            specialist_results or {}
        ).items():

            specialist_blocks.append(
                f"""
==============================
{agent_name}
==============================

{self._review_to_text(review)}
"""
            )

        specialist_text = (
            "\n\n".join(specialist_blocks)
            if specialist_blocks
            else "No specialist reviews were supplied."
        )

        verification_text = self._verification_to_text(
            verification_results
        )

        prompt = f"""
You are the Final Reviewer Agent in AutoReview,
an automated research-paper review system.

Your task is to consolidate seven specialist reviews into
a useful, consistent final assessment.

IMPORTANT RULES:

1. Preserve important strengths and weaknesses.

2. Do not invent manuscript details, sources, citations,
   experiments, results, or numerical values.

3. Do not present an unverified claim as an established fact.

4. Findings marked SUPPORTED may be described as supported
   by the supplied evidence. This does not guarantee absolute truth.

5. Findings marked NEEDS_REVISION must not be presented as
   established facts. Describe them as concerns that need correction
   or additional evidence.

6. Findings marked UNVERIFIED must not be presented as verified.
   If important, include them in UNVERIFIED CONCERNS.

7. If verification results are missing, explicitly state that
   independent evidence verification was not provided.

8. If the verifier failed because of an API or quota error,
   report the limitation. Do not conclude that the specialist's
   finding is false merely because verification failed.

9. Identify disagreements between specialist agents instead of
   silently resolving them.

10. Combine repeated findings where appropriate.

11. Give specific and practical recommendations.

12. The overall score below was calculated by the existing
    specialist scoring system. Do not change it.

CALCULATED SPECIALIST SCORE:
{overall_score}/10

SPECIALIST REVIEWS:
{specialist_text[:18000]}

VERIFICATION RESULTS:
{verification_text[:9000]}

Return the final review in exactly this format:

==================================================
FINAL CONSOLIDATED REVIEW
==================================================

OVERALL SCORE:
{overall_score}/10

VERIFICATION SUMMARY:
- Summarize the verification results supplied.
- State clearly if verification results were unavailable.

EXECUTIVE SUMMARY:
- Point 1
- Point 2
- Point 3

MAJOR STRENGTHS:
- Point 1
- Point 2
- Point 3

MAJOR WEAKNESSES:
- Point 1
- Point 2
- Point 3

STRUCTURAL ISSUES:
- Point 1
- Point 2

NOVELTY AND RESEARCH GAP:
- Point 1
- Point 2

METHODOLOGY ISSUES:
- Point 1
- Point 2

EXPERIMENTAL ISSUES:
- Point 1
- Point 2

LITERATURE AND CITATION ISSUES:
- Point 1
- Point 2

WRITING ISSUES:
- Point 1
- Point 2

UNVERIFIED CONCERNS:
- List important concerns requiring additional evidence.
- If none are identified, say so explicitly.

PRIORITY RECOMMENDATIONS:
1. Recommendation
2. Recommendation
3. Recommendation
4. Recommendation
5. Recommendation

FINAL ASSESSMENT:
- Provide a concise assessment distinguishing the specialist score
  from the confidence provided by evidence verification.
"""

        return prompt

    def review(
        self,
        paper_text,
        specialist_results,
        overall_score,
        verification_results=None,
    ):

        if not specialist_results:
            return (
                "Final Reviewer could not run because "
                "no specialist agent results were available."
            )

        prompt = self.build_prompt(
            paper_text=paper_text,
            specialist_results=specialist_results,
            overall_score=overall_score,
            verification_results=verification_results,
        )

        try:
            response = ask_qwen(
                prompt,
                max_tokens=1800,
            )

            if not response or not str(response).strip():
                return (
                    "Final Reviewer returned an empty response. "
                    "Check the model service and API quota."
                )

            return response

        except Exception as error:
            return (
                "Final Reviewer could not complete the consolidated "
                f"review ({type(error).__name__}). Check the model "
                "service, internet connection, and API quota. "
                "Preserve the specialist results and verification "
                "statuses from this run."
            )