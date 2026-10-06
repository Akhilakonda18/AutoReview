from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class WritingAgent(BaseAgent):

    def __init__(self):
        super().__init__("Writing Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):

        prompt = f"""
You are the Writing Quality Agent in an automated research
paper review system.

Evaluate the technical writing quality of the research paper.

Analyze:

1. Clarity of writing
2. Grammar and sentence construction
3. Technical terminology
4. Readability
5. Logical flow between sections
6. Consistency of terminology
7. Ambiguous or unclear statements
8. Overall academic writing quality

Do not focus on whether the research contribution is novel.
Focus specifically on how clearly and professionally the
research is communicated.

Give a WRITING SCORE from 0 to 10.

Research Paper:
{paper_text}

Retrieved Evidence:
{retrieved_evidence}

Return the result in exactly this format:

WRITING SCORE: X/10

STRENGTHS:
- point 1
- point 2

WEAKNESSES:
- point 1
- point 2

CLARITY ISSUES:
- point 1
- point 2

TECHNICAL WRITING ISSUES:
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

    agent = WritingAgent()

    sample_paper = """
    Artificial intelligence is increasingly used in
    scientific research. The proposed system evaluates
    research papers using language models and retrieval.
    """

    result = agent.review(sample_paper)

    print("=" * 60)
    print("WRITING AGENT")
    print("=" * 60)

    print(result["review"])