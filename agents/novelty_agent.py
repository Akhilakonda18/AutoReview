from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class NoveltyAgent(BaseAgent):

    def __init__(self):
        super().__init__("Novelty Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Novelty Agent in an automated research paper
review system.

Your task is to evaluate the originality and novelty of the
research paper using the paper itself and the retrieved
research evidence.

Evaluate:

1. Main research contribution
2. Originality of the proposed approach
3. Similarity with existing research
4. Difference from retrieved papers
5. Potential research gap
6. Whether the claimed contribution appears sufficiently novel

Give a NOVELTY SCORE from 0 to 10.

Important:
Do not claim that a paper is definitely novel or non-novel
based only on the retrieved papers. Explain the evidence
and limitations of the assessment.

Retrieved Research Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

NOVELTY SCORE: X/10

MAIN CONTRIBUTION:
- point 1
- point 2

NOVELTY STRENGTHS:
- point 1
- point 2

SIMILAR OR RELATED WORK:
- point 1
- point 2

POTENTIAL RESEARCH GAP:
- point 1
- point 2

NOVELTY WEAKNESSES:
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

    agent = NoveltyAgent()

    sample_paper = """
    This research paper proposes a new framework for
    automated research paper review using LLMs and RAG.
    """

    sample_evidence = """
    Previous research has explored LLM-based automated
    peer review and multi-agent paper review systems.
    """

    result = agent.review(
        sample_paper,
        sample_evidence
    )

    print("=" * 60)
    print("NOVELTY AGENT")
    print("=" * 60)

    print(result["review"])