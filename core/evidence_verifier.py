
import json
import re

from core.llm_service import ask_qwen


NO_REFERENCE_EVIDENCE = "No reference evidence was retrieved."


def _evidence_to_text(evidence, max_chars=9000):
    """Convert retrieved evidence into readable, source-labelled text."""

    if evidence is None or evidence == [] or evidence == {} or evidence == "":
        return NO_REFERENCE_EVIDENCE

    if isinstance(evidence, str):
        return evidence[:max_chars]

    if isinstance(evidence, dict):
        evidence = [evidence]

    parts = []

    for index, item in enumerate(evidence, start=1):
        if isinstance(item, dict):
            source = (
                item.get("filename")
                or item.get("title")
                or item.get("source")
                or "Unknown source"
            )

            content = (
                item.get("text")
                or item.get("content")
                or item.get("chunk")
                or item.get("page_content")
                or ""
            )

            similarity = item.get("similarity")

            metadata = (
                f" | similarity={similarity}"
                if similarity is not None
                else ""
            )

        else:
            source = f"Evidence {index}"
            content = str(item)
            metadata = ""

        if content:
            parts.append(
                f"[Evidence {index} | Source: {source}{metadata}]\n"
                f"{content}"
            )

    if not parts:
        return NO_REFERENCE_EVIDENCE

    return "\n\n".join(parts)[:max_chars]


def _extract_json_object(response):
    """Extract a JSON object from the model response."""

    if not response or not str(response).strip():
        raise ValueError(
            "Evidence verifier returned an empty response."
        )

    cleaned = str(response).strip()

    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(
                cleaned[index:]
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse the evidence verifier's JSON response."
    )


def _as_string_list(value):
    """Normalize a value into a list of non-empty strings."""

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if str(value).strip():
        return [str(value).strip()]

    return []


def _unverified(
    feedback,
    unsupported_claims=None,
    missing_evidence=None,
):
    """Return a safe result when verification cannot be completed."""

    return {
        "status": "UNVERIFIED",
        "unsupported_claims": _as_string_list(
            unsupported_claims
        ),
        "missing_evidence": _as_string_list(
            missing_evidence
        ),
        "feedback": str(
            feedback or "Verification could not be completed."
        ),
    }


def verify_review(
    review_text,
    evidence,
    paper_text=None,
    agent_name=None,
):
    """
    Verify a specialist review against manuscript text and external evidence.

    Backwards-compatible with:
        verify_review(review_text, evidence)

    Optional arguments:
        paper_text: text extracted from the submitted manuscript.
        agent_name: name of the specialist agent.

    Possible statuses:
        SUPPORTED
        NEEDS_REVISION
        UNVERIFIED
    """

    review_text = str(review_text or "").strip()
    manuscript_text = str(paper_text or "").strip()

    evidence_text = _evidence_to_text(evidence)

    if not review_text:
        return {
            "status": "NEEDS_REVISION",
            "unsupported_claims": [
                "The specialist review is empty."
            ],
            "missing_evidence": [
                "A non-empty specialist review is required."
            ],
            "feedback": (
                "Generate the review before verification."
            ),
        }

    if (
        not manuscript_text
        and evidence_text == NO_REFERENCE_EVIDENCE
    ):
        return _unverified(
            (
                "Neither manuscript text nor retrieved "
                "reference evidence was supplied."
            ),
            missing_evidence=[
                (
                    "Supply the manuscript text and/or "
                    "relevant external research evidence."
                )
            ],
        )

    if not manuscript_text:
        manuscript_text = (
            "MANUSCRIPT TEXT NOT PROVIDED. Claims about the "
            "submitted paper's own sections, methods, datasets, "
            "and results cannot be checked against the manuscript."
        )

    if evidence_text == NO_REFERENCE_EVIDENCE:
        evidence_text = (
            "NO EXTERNAL REFERENCE EVIDENCE WAS RETRIEVED. "
            "Claims about prior research, the state of the art, "
            "and novelty cannot be verified against external sources."
        )

    prompt = f"""
You are the evidence verifier for AutoReview,
an automated research-paper review system.

SPECIALIST AGENT:
{agent_name or "Unspecified specialist agent"}

TASK:
Evaluate the material factual claims in the specialist review
against the appropriate supplied source.

SOURCE RULES:

1. Claims about what the submitted manuscript contains, implements,
   measures, or reports must be checked against the manuscript text.

2. Claims about prior research, comparisons with existing methods,
   research gaps, novelty, and the state of the art must be checked
   against external research evidence.

3. Do not use external papers to prove what the submitted manuscript
   itself contains.

4. Do not use the manuscript alone to establish claims about the
   entire research field.

5. If the relevant source is missing, do not mark the claim as verified.
   Explain what additional evidence is needed.

6. Editorial suggestions do not all require external citations.
   Evaluate whether material factual findings have a reasonable basis.

7. Do not assume a claim is supported merely because it sounds plausible.

8. Do not invent facts, quotations, sources, or numerical values.

9. Return one valid JSON object only.

10. Use SUPPORTED only when material factual claims are adequately
    supported by the appropriate supplied sources.

11. Use NEEDS_REVISION when material claims are contradicted or
    unsupported by the available evidence.

12. Use UNVERIFIED when the available material is insufficient to
    complete a meaningful check.

SPECIALIST REVIEW:
{review_text[:10000]}

SUBMITTED MANUSCRIPT TEXT:
{manuscript_text[:14000]}

EXTERNAL RESEARCH EVIDENCE:
{evidence_text[:9000]}

Return exactly this JSON structure:

{{
  "status": "SUPPORTED",
  "unsupported_claims": [],
  "missing_evidence": [],
  "feedback": "Concise explanation of the verification decision"
}}

When material claims cannot be checked because a source is missing,
use UNVERIFIED rather than assuming the claims are false or supported.
"""

    try:
        response = ask_qwen(
            prompt,
            max_tokens=900,
        )

    except Exception as error:
        return _unverified(
            (
                "Verifier API call failed with "
                f"{type(error).__name__}. Check the model service, "
                "internet connection, and API quota."
            )
        )

    try:
        result = _extract_json_object(response)

    except ValueError as error:
        return _unverified(str(error))

    status = str(
        result.get("status", "")
    ).strip().upper()

    unsupported_claims = _as_string_list(
        result.get("unsupported_claims")
    )

    missing_evidence = _as_string_list(
        result.get("missing_evidence")
    )

    feedback = str(
        result.get("feedback", "")
    ).strip()

    if status not in {
        "SUPPORTED",
        "NEEDS_REVISION",
        "UNVERIFIED",
    }:
        return _unverified(
            (
                "Verifier returned an invalid status. "
                "The review has not been verified."
            ),
            unsupported_claims,
            missing_evidence,
        )

    # A SUPPORTED response cannot simultaneously list unsupported claims.
    if status == "SUPPORTED" and unsupported_claims:
        status = "NEEDS_REVISION"

        feedback = (
            feedback + " "
            + "Status changed to NEEDS_REVISION because "
            "unsupported claims were listed."
        ).strip()

    return {
        "status": status,
        "unsupported_claims": unsupported_claims,
        "missing_evidence": missing_evidence,
        "feedback": feedback,
    }
