from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class ReviewerAgent(BaseAgent):

    def __init__(self):

        super().__init__(
            "Final Reviewer Agent"
        )

    def build_prompt(
        self,
        paper_text,
        agent_results,
        agent_scores,
        overall_score
    ):

        prompt = f"""
You are the Final Reviewer Agent in an automated
research paper review system.

You receive evaluations from seven specialized
review agents.

Your task is to consolidate their findings into
one final research paper review.

The specialized agents are:

1. Structure Agent
2. Novelty Agent
3. Methodology Agent
4. Literature Agent
5. Experiment Agent
6. Writing Agent
7. Citation Agent

IMPORTANT INSTRUCTIONS:

- Base your assessment only on the research paper
  and specialized agent results.
- Do not invent facts.
- Do not blindly agree with the agents.
- Identify important disagreements.
- Give more attention to repeated issues.
- Distinguish major issues from minor issues.
- Keep the final review concise but informative.
- The overall score below is calculated by Score Fusion.
- Do not change the score.

--------------------------------------------------
FUSED AGENT SCORES
--------------------------------------------------

{agent_scores}

--------------------------------------------------
FUSED OVERALL SCORE
--------------------------------------------------

{overall_score}/10

--------------------------------------------------
SPECIALIZED AGENT REVIEWS
--------------------------------------------------

{agent_results}

--------------------------------------------------
RESEARCH PAPER
--------------------------------------------------

{paper_text}

--------------------------------------------------
FINAL REVIEW FORMAT
--------------------------------------------------

Return exactly this format:

OVERALL SCORE: {overall_score}/10

OVERALL STRENGTHS:
- point 1
- point 2
- point 3

OVERALL WEAKNESSES:
- point 1
- point 2
- point 3

MAJOR ISSUES:
- point 1
- point 2

MINOR ISSUES:
- point 1
- point 2

AGENT DISAGREEMENTS:
- point 1
- point 2

PRIORITY IMPROVEMENTS:
1. improvement 1
2. improvement 2
3. improvement 3

FINAL RECOMMENDATION:
- Accept
- Minor Revision
- Major Revision
- Reject

REASON:
Explain the recommendation briefly based only
on the evidence provided by the paper and agents.
"""

        return prompt

    def review(
        self,
        paper_text,
        agent_results,
        agent_scores=None,
        overall_score=None
    ):

        if agent_scores is None:
            agent_scores = {}

        if overall_score is None:
            overall_score = 0

        prompt = self.build_prompt(
            paper_text,
            agent_results,
            agent_scores,
            overall_score
        )

        try:

            # IMPORTANT:
            # Do not pass max_retries here.
            # ask_qwen() does not accept it.

            response = ask_qwen(
                prompt,
                max_tokens=1800
            )

            return {
                "agent": self.name,
                "review": response
            }

        except Exception as e:

            print(
                "\nERROR while running "
                "Final Reviewer:"
            )

            print(e)

            return {
                "agent": self.name,
                "review":
                    "Final Reviewer Agent failed.\n\n"
                    f"Error: {str(e)}"
            }