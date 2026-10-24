
import re

from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class ReviewerAgent(BaseAgent):

    def __init__(self):
        super().__init__("Final Reviewer Agent")

    def build_prompt(
        self,
        paper_text,
        agent_results,
        agent_scores,
        overall_score,
    ):
        score_text = "\n".join(
            f"- {name}: {score}/10"
            for name, score in agent_scores.items()
        )

        # Normalize specialist results for the prompt.
        reviews_text = "\n\n".join(
            f"===== {name} =====\n{review}"
            for name, review in agent_results.items()
        )

        prompt = f"""
You are the Final Reviewer Agent in an automated
research paper review system.

Consolidate the seven specialist reviews into one
evidence-based final research paper review.

IMPORTANT SCORING RULES:
1. The overall score is calculated externally by
   the score-fusion stage.
2. The authoritative overall score is {overall_score}/10.
3. You MUST reproduce this exact score in your output.
4. Do not calculate, estimate, or change the overall score.
5. Do not use 0/10 unless the supplied fused score is 0.
6. Individual scores are provided for transparency.

REVIEW RULES:
- Base your assessment on the paper and specialist reviews.
- Do not invent facts, references, experiments, or results.
- Identify meaningful disagreements between agents.
- Give more attention to issues identified repeatedly.
- Distinguish major issues from minor issues.
- Make recommendations specific and actionable.
- Do not treat retrieved evidence as proof unless relevant.
- If evidence is insufficient, state the limitation.

FUSED INDIVIDUAL SCORES:
{score_text}

AUTHORITATIVE FUSED OVERALL SCORE:
{overall_score}/10

SPECIALIST REVIEWS:
{reviews_text}

RESEARCH PAPER:
{paper_text}

Return the following sections:

OVERALL SCORE: {overall_score}/10

OVERALL STRENGTHS:
- Three key strengths

OVERALL WEAKNESSES:
- Three key weaknesses

MAJOR ISSUES:
- Important issues requiring substantial attention

MINOR ISSUES:
- Smaller issues that can be corrected

AGENT DISAGREEMENTS:
- Important disagreements between specialists
- If there are none, state that explicitly

PRIORITY IMPROVEMENTS:
1. Highest-priority improvement
2. Second improvement
3. Third improvement

FINAL RECOMMENDATION:
Choose exactly one: Accept, Minor Revision,
Major Revision, or Reject.

REASON:
Briefly justify the recommendation using the available
paper findings and specialist reviews.
"""
        return prompt

    def review(
        self,
        paper_text,
        agent_results,
        agent_scores=None,
        overall_score=None,
    ):
        agent_scores = agent_scores or {}

        # Also support receiving the complete score dictionary
        # from review_pipeline.py.
        if isinstance(agent_scores, dict) and (
            "individual_scores" in agent_scores
            or "overall_score" in agent_scores
        ):
            score_data = agent_scores
            agent_scores = score_data.get(
                "individual_scores", {}
            )

            if overall_score is None:
                overall_score = score_data.get("overall_score")

        if overall_score is None:
            raise ValueError(
                "The fused overall score was not supplied "
                "to ReviewerAgent."
            )

        overall_score = round(float(overall_score), 2)

        prompt = self.build_prompt(
            paper_text,
            agent_results,
            agent_scores,
            overall_score,
        )

        try:
            response = ask_qwen(
                prompt,
                max_tokens=1800,
            )

            if not response:
                raise ValueError(
                    "The language model returned an empty review."
                )

            response = str(response).strip()

            # Enforce the authoritative fused score even if
            # the language model generates a conflicting score.
            score_line = f"OVERALL SCORE: {overall_score:.2f}/10"

            if re.search(
                r"^\s*OVERALL\s+SCORE\s*:.*$",
                response,
                flags=re.IGNORECASE | re.MULTILINE,
            ):
                response = re.sub(
                    r"^\s*OVERALL\s+SCORE\s*:.*$",
                    score_line,
                    response,
                    count=1,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            else:
                response = score_line + "\n\n" + response

            return {
                "agent": self.name,
                "review": response,
            }

        except Exception as exc:
            print("\nERROR while running Final Reviewer:")
            print(exc)

            return {
                "agent": self.name,
                "review": (
                    "Final Reviewer Agent failed.\n\n"
                    f"Error: {exc}"
                ),
            }
