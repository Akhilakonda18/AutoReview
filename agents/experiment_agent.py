from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class ExperimentAgent(BaseAgent):

    def __init__(self):
        super().__init__("Experiment Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Experiment Evaluation Agent in an automated
research paper review system.

Evaluate the experimental quality of the research paper.

Analyze:

1. Dataset description and suitability
2. Experimental setup
3. Baseline comparisons
4. Evaluation metrics
5. Results and interpretation
6. Ablation studies
7. Statistical analysis
8. Reproducibility of experiments
9. Limitations of the experimental evaluation

Give an EXPERIMENT SCORE from 0 to 10.

Do not invent experimental results that are not present
in the paper.

Retrieved Research Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

EXPERIMENT SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

EXPERIMENTAL GAPS:
- point 1
- point 2

BASELINE EVALUATION:
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

    agent = ExperimentAgent()

    sample_paper = """
    This paper proposes an AI-based research paper review
    framework. The system is evaluated using a dataset of
    research papers and several evaluation metrics.
    """

    sample_evidence = """
    Existing research paper review systems commonly compare
    their methods against baseline approaches and report
    evaluation metrics.
    """

    result = agent.review(
        sample_paper,
        sample_evidence
    )

    print("=" * 60)
    print("EXPERIMENT AGENT")
    print("=" * 60)

    print(result["review"])