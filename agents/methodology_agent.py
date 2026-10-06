from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class MethodologyAgent(BaseAgent):

    def __init__(self):
        super().__init__("Methodology Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Methodology Agent in an automated research
paper review system.

Evaluate the methodology of the research paper.

Analyze:

1. Research methodology
2. Proposed algorithms or models
3. Dataset and data preparation
4. Experimental procedure
5. Baseline methods
6. Evaluation metrics
7. Reproducibility
8. Methodological limitations

Give a METHODOLOGY SCORE from 0 to 10.

Retrieved Research Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

METHODOLOGY SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

METHODOLOGICAL GAPS:
- point 1
- point 2

REPRODUCIBILITY:
- point 1
- point 2

SUGGESTIONS:
- point 1
- point 2

EVIDENCE:
- point 1
- point 2
"""

        return prompt

    def review(self, paper_text, retrieved_evidence=""):

        prompt = self.build_prompt(
            paper_text,
            retrieved_evidence
        )

        response = ask_qwen(prompt)

        return {
            "agent": self.name,
            "review": response
        }


if __name__ == "__main__":

    agent = MethodologyAgent()

    sample_paper = """
    This paper proposes a machine learning framework
    for automated research paper evaluation.
    The model is evaluated using a research paper dataset.
    """

    sample_evidence = """
    Existing research uses different machine learning
    and language-model-based approaches for paper review.
    """

    result = agent.review(
        sample_paper,
        sample_evidence
    )

    print("=" * 60)
    print("METHODOLOGY AGENT")
    print("=" * 60)

    print(result["review"])