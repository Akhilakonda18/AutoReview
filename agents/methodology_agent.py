from agents.base_agent import BaseAgent
from core.llm_service import ask_qwen


class MethodologyAgent(BaseAgent):
    def __init__(self):
        super().__init__("Methodology Agent")

    def build_prompt(self, paper_text, retrieved_evidence=""):
        return f"""
You are the Methodology Agent in an automated research-paper review system.
Evaluate the described research design, algorithm/model, dataset and preprocessing, experimental procedure, baselines, metrics, assumptions, limitations, and reproducibility. Do not infer unreported datasets, hyperparameters, metrics, or procedures. Retrieved papers may provide comparison context but cannot establish what this manuscript implemented.

Rules:
- Treat the submitted manuscript as the primary source for claims about its own content.
- Treat retrieved papers only as external comparison/context; they cannot prove what this manuscript did.
- Never invent facts, citations, quotations, numbers, results, headings, or implementation details.
- If information is absent from the supplied text, say "not reported in the supplied manuscript."
- Distinguish "not found in retrieved evidence" from "does not exist."
- If the supplied text or evidence is insufficient, explicitly state the limitation and avoid definitive claims.
- Give a score from 0 to 10 for methodological clarity, rigor, and reproducibility.
- Provide specific, actionable observations anchored in the supplied text/evidence.

Retrieved External Evidence:
{retrieved_evidence or "No external evidence supplied."}

Submitted Manuscript:
{paper_text or "[No manuscript text supplied]"}

Return exactly this format:

METHODOLOGY SCORE: X/10

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
    result = MethodologyAgent().review("Sample manuscript text for a basic smoke test.")
    print(result["review"])
