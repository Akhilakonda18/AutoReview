from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class NoveltyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Novelty Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):
        return f"""
You are the Novelty Agent in an automated research-paper review system.
Identify the contribution claimed in the manuscript, then compare it with the supplied retrieved research. Do not claim comprehensive novelty or assert that no prior work exists based on a limited retrieval set. Describe comparisons only when supported by supplied titles/excerpts. Label the score as strength and substantiation of the novelty claim, not proof of absolute originality.

Rules:
- Treat the submitted manuscript as the primary source for claims about its own content.
- Treat retrieved papers only as external comparison/context; they cannot prove what this manuscript did.
- Never invent facts, citations, quotations, numbers, results, headings, or implementation details.
- If information is absent from the supplied text, say "not reported in the supplied manuscript."
- Distinguish "not found in retrieved evidence" from "does not exist."
- If the supplied text or evidence is insufficient, explicitly state the limitation and avoid definitive claims.
- Give a score from 0 to 10 for strength and substantiation of the novelty claim.
- Provide specific, actionable observations anchored in the supplied text/evidence.

Retrieved External Evidence:
{retrieved_evidence or "No external evidence supplied."}

Submitted Manuscript:
{paper_text or "[No manuscript text supplied]"}

Return exactly this format:

NOVELTY SCORE: X/10

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
    result = NoveltyAgent().review("Sample manuscript text for a basic smoke test.")
    print(result["review"])
