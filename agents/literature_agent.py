from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class LiteratureAgent(BaseAgent):

    def __init__(self):
        super().__init__("Literature Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Literature Review Agent in an automated
research paper review system.

Evaluate the literature review and related work of the
research paper.

Analyze:

1. Coverage of existing research
2. Relevance of cited studies
3. Organization of related work
4. Identification of research gaps
5. Use of recent and important references
6. Connection between previous work and the proposed work
7. Missing relevant literature

Use the retrieved research papers as supporting evidence.

Do not claim that a reference is definitely missing unless
the available evidence supports that conclusion.

Give a LITERATURE SCORE from 0 to 10.

Retrieved Research Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

LITERATURE SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

MISSING LITERATURE AREAS:
- point 1
- point 2

RESEARCH GAPS:
- point 1
- point 2

RECOMMENDED REFERENCES:
- topic or paper area 1
- topic or paper area 2

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

    agent = LiteratureAgent()

    sample_paper = """
    This paper proposes an automated research paper review
    framework using Large Language Models and RAG.
    The related work discusses existing LLM-based review
    systems.
    """

    sample_evidence = """
    Existing research has explored automated peer review,
    LLM-based paper evaluation, RAG, and multi-agent review.
    """

    result = agent.review(
        sample_paper,
        sample_evidence
    )

    print("=" * 60)
    print("LITERATURE AGENT")
    print("=" * 60)

    print(result["review"])