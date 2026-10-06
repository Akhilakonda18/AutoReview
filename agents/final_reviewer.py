from core.llm_service import ask_qwen


class FinalReviewer:

    def __init__(self):
        self.name = "Final Reviewer Agent"

    def build_prompt(
        self,
        paper_text,
        specialist_results,
        overall_score
    ):

        # Keep the paper itself out of the final prompt.
        # The specialist agents have already analyzed it.
        #
        # This prevents the prompt from becoming too large.

        specialist_text = ""

        for agent_name, review in specialist_results.items():

            specialist_text += f"""

==============================
{agent_name}
==============================

{review}

"""

        prompt = f"""
You are the Final Reviewer Agent in an automated
research paper review system.

Your task is to consolidate the outputs of seven
specialist review agents into one final research
paper review.

The specialist agents are:

1. Structure Agent
2. Novelty Agent
3. Methodology Agent
4. Literature Agent
5. Experiment Agent
6. Writing Agent
7. Citation Agent

IMPORTANT RULES:

- Do not invent information.
- Use only the specialist reviews provided below.
- Preserve important weaknesses identified by agents.
- Preserve important strengths identified by agents.
- Identify common problems appearing across multiple agents.
- Identify conflicting observations if they exist.
- Give practical recommendations.
- Do not repeat the entire specialist reviews.
- Produce a concise but complete consolidated review.

The calculated average score from the specialist agents is:

OVERALL SCORE: {overall_score}/10

SPECIALIST REVIEWS:
{specialist_text}

Return the final review in exactly this format:

==================================================
FINAL CONSOLIDATED REVIEW
==================================================

OVERALL SCORE:
{overall_score}/10

EXECUTIVE SUMMARY:
- point 1
- point 2
- point 3

MAJOR STRENGTHS:
- point 1
- point 2
- point 3

MAJOR WEAKNESSES:
- point 1
- point 2
- point 3

STRUCTURAL ISSUES:
- point 1
- point 2

NOVELTY AND RESEARCH GAP:
- point 1
- point 2

METHODOLOGY ISSUES:
- point 1
- point 2

EXPERIMENTAL ISSUES:
- point 1
- point 2

LITERATURE AND CITATION ISSUES:
- point 1
- point 2

WRITING ISSUES:
- point 1
- point 2

PRIORITY RECOMMENDATIONS:
1. recommendation
2. recommendation
3. recommendation
4. recommendation
5. recommendation

FINAL ASSESSMENT:
- concise overall assessment
"""

        return prompt

    def review(
        self,
        paper_text,
        specialist_results,
        overall_score
    ):

        if not specialist_results:

            return (
                "Final Reviewer could not run because "
                "no specialist agent results were available."
            )

        prompt = self.build_prompt(
            paper_text,
            specialist_results,
            overall_score
        )

        # IMPORTANT:
        # Do NOT use max_retries here.
        # Your ask_qwen() function does not accept it.

        response = ask_qwen(
            prompt,
            max_tokens=1500
        )

        return response