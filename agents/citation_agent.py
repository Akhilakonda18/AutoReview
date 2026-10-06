from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class CitationAgent(BaseAgent):

    def __init__(self):
        super().__init__("Citation Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Citation Verification Agent in an automated
research paper review system.

Evaluate the citation and reference quality of the research
paper using only the information available in the paper and
the retrieved research evidence.

Analyze:

1. Citation consistency
2. Citation coverage of important claims
3. Relevance of cited literature
4. References related to the research topic
5. Potentially unsupported claims
6. Potential missing literature
7. Reference quality and completeness

Important:
Do not invent citation numbers, authors, titles, or papers.
If the available information is insufficient to verify a
citation, clearly state that it cannot be verified.

Give a CITATION SCORE from 0 to 10.

Retrieved Research Evidence:
{retrieved_evidence}

Research Paper:
{paper_text}

Return the result in exactly this format:

CITATION SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

POTENTIALLY UNSUPPORTED CLAIMS:
- point 1
- point 2

MISSING LITERATURE AREAS:
- point 1
- point 2

RECOMMENDED REFERENCE TOPICS:
- topic 1
- topic 2

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

    agent = CitationAgent()

    sample_paper = """
    This research paper proposes a new AI framework for
    automated research paper review using large language
    models and retrieval augmented generation.
    """

    sample_evidence = """
    Previous research has explored automated peer review,
    LLM-based paper evaluation, RAG, and multi-agent systems.
    """

    result = agent.review(
        sample_paper,
        sample_evidence
    )

    print("=" * 60)
    print("CITATION AGENT")
    print("=" * 60)

    print(result["review"])