
import json
import re

from core.llm_service import ask_qwen


def _evidence_to_text(evidence, max_chars=9000):
    """Convert retrieved evidence objects into readable text."""

    if evidence is None or evidence == [] or evidence == {} or evidence == "":
        return "No reference evidence was retrieved."

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
        else:
            source = f"Evidence {index}"
            content = str(item)

        if content:
            parts.append(
                f"[Evidence {index} | Source: {source}]\n{content}"
            )

    if not parts:
        return "No reference evidence was retrieved."

    return "\n\n".join(parts)[:max_chars]


def _extract_json_object(response):
    """Parse JSON, tolerating Markdown fences and surrounding text."""

    if not response or not str(response).strip():
        raise ValueError(
            "Evidence verifier returned an empty response."
        )

    cleaned = str(response).strip()

    # Remove optional Markdown code fences.
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    # First try parsing the complete response.
    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # If extra text exists, find the first valid JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(cleaned[index:])

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse the evidence verifier's JSON response."
    )


def _as_string_list(value):
    """Normalize response fields into lists of non-empty strings."""

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


def verify_review(review_text, evidence):
    """
    Check whether a specialist review is supported by retrieved evidence.

    Returns:
        status
        unsupported_claims
        missing_evidence
        feedback

    This LLM-based check cannot guarantee factual correctness.
    """

    review_text = str(review_text or "").strip()
    evidence_text = _evidence_to_text(evidence)

    # Handle an empty review.
    if not review_text:
        return {
            "status": "NEEDS_REVISION",
            "unsupported_claims": [
                "The review is empty."
            ],
            "missing_evidence": [
                "A non-empty specialist review is required."
            ],
            "feedback": (
                "Generate the review before verification."
            ),
        }

    # Handle missing evidence.
    if evidence_text.startswith(
        "No reference evidence was retrieved."
    ):
        return {
            "status": "NEEDS_REVISION",
            "unsupported_claims": [
                "The review cannot be checked against reference evidence."
            ],
            "missing_evidence": [
                "Retrieve relevant reference evidence for the review."
            ],
            "feedback": (
                "Retrieve relevant sources. Clearly distinguish "
                "paper-specific observations from claims requiring "
                "external evidence."
            ),
        }

    prompt = f"""
You are an evidence-verification component in AutoReview,
an automated research-paper review system.

TASK:
Check whether the specialist review below is supported by the
provided retrieved reference evidence.

IMPORTANT RULES:
1. Do not assume a claim is supported just because it sounds plausible.
2. A related reference does not automatically support a specific claim.
3. Distinguish claims about the submitted paper from claims about prior research.
4. Identify factual claims, comparisons, and numerical statements
   that lack adequate support.
5. Do not require every sentence to have a citation.
6. Do not invent sources, facts, or evidence.
7. Judge support using the supplied evidence only.
8. If evidence is insufficient, request specific additional evidence.
9. Return one valid JSON object only.
10. The status must be SUPPORTED or NEEDS_REVISION.

SPECIALIST REVIEW:
{review_text[:10000]}

RETRIEVED REFERENCE EVIDENCE:
{evidence_text}

Return this JSON structure:
{{
  "status": "SUPPORTED",
  "unsupported_claims": [],
  "missing_evidence": [],
  "feedback": "Concise explanation"
}}

Use SUPPORTED only when important evidence-dependent claims are
adequately supported.

If material claims are unsupported or the evidence is insufficient,
use NEEDS_REVISION.

When the status is SUPPORTED, use empty arrays for the two list
fields unless there is a specific unresolved issue.
"""

    # Ask the existing Gemini service to verify the review.
    response = ask_qwen(prompt, max_tokens=900)

    # Parse the response without crashing on malformed JSON.
    try:
        result = _extract_json_object(response)

    except ValueError as error:
        return {
            "status": "UNVERIFIED",
            "unsupported_claims": [],
            "missing_evidence": [],
            "feedback": str(error),
        }

    status = str(
        result.get("status", "")
    ).strip().upper()

    # Do not treat an invalid status as a successful verification.
    if status not in {"SUPPORTED", "NEEDS_REVISION"}:
        return {
            "status": "UNVERIFIED",
            "unsupported_claims": _as_string_list(
                result.get("unsupported_claims")
            ),
            "missing_evidence": _as_string_list(
                result.get("missing_evidence")
            ),
            "feedback": (
                "Verifier returned an invalid status. "
                + str(result.get("feedback", "")).strip()
            ).strip(),
        }

    return {
        "status": status,
        "unsupported_claims": _as_string_list(
            result.get("unsupported_claims")
        ),
        "missing_evidence": _as_string_list(
            result.get("missing_evidence")
        ),
        "feedback": str(
            result.get("feedback", "")
        ).strip(),
    }
