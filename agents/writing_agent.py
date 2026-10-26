from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class WritingAgent(BaseAgent):
    def __init__(self):
        super().__init__("Writing Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):
        return f"""
You are the Writing Quality Agent in an automated research-paper review system.
Evaluate only the manuscript's writing: grammar, clarity, concision, academic tone, terminology, readability, transitions, ambiguity, repetition, and visible figure/table references. Do not judge novelty or use external papers as a writing-quality benchmark. Quote short phrases only when they genuinely appear in the supplied text.

Rules:
- Treat the submitted manuscript as the primary source for claims about its own content.
- Treat retrieved papers only as external comparison/context; they cannot prove what this manuscript did.
- Never invent facts, citations, quotations, numbers, results, headings, or implementation details.
- If information is absent from the supplied text, say "not reported in the supplied manuscript."
- Distinguish "not found in retrieved evidence" from "does not exist."
- If the supplied text or evidence is insufficient, explicitly state the limitation and avoid definitive claims.
- Give a score from 0 to 10 for writing clarity and academic presentation.
- Provide specific, actionable observations anchored in the supplied text/evidence.

Retrieved External Evidence:
{retrieved_evidence or "No external evidence supplied."}

Submitted Manuscript:
{paper_text or "[No manuscript text supplied]"}

Return exactly this format:

WRITING SCORE: X/10

STRENGTHS:
- ...

WEAKNESSES:
- ...

KEY FINDINGS / GAPS:
- ...

SUGGESTIONS:
- ...

EVIDENCE AND LIMITATIONS:
- ...
"""

    def review(self, paper_text, retrieved_evidence=""):
        response = ask_qwen(self.build_prompt(paper_text, retrieved_evidence))
        return {"agent": self.name, "review": response}


if __name__ == "__main__":
    result = WritingAgent().review("Sample manuscript text for a basic smoke test.")
    print(result["review"])
