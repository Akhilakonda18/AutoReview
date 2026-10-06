from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class StructureAgent(BaseAgent):

    def __init__(self):
        super().__init__("Structure Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Structure Agent in an automated research paper
review system.

Your task is to evaluate the structure and organization of
the research paper.

Evaluate:

1. Abstract
2. Introduction
3. Related Work
4. Methodology
5. Experiments
6. Results
7. Discussion
8. Conclusion
9. References
10. Overall organization

Give a score from 0 to 10.

Identify:

- Strengths
- Weaknesses
- Missing or unclear sections
- Suggestions for improvement

Use the retrieved research evidence only when it is relevant.

Retrieved Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

MISSING OR UNCLEAR SECTIONS:
- point 1
- point 2

SUGGESTIONS:
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

    agent = StructureAgent()

    sample_paper = """
    This is a sample research paper about
    Large Language Models and automated peer review.
    """

    result = agent.review(sample_paper)

    print("=" * 60)
    print("STRUCTURE AGENT")
    print("=" * 60)

    print(result["review"])