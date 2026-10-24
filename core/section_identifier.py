
import re


# =========================================================
# CANONICAL RESEARCH PAPER SECTION NAMES
# =========================================================

SECTION_PATTERNS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "literature review": "Related Work",
    "background": "Background",
    "methodology": "Methodology",
    "method": "Methodology",
    "methods": "Methodology",
    "experimental setup": "Experimental Setup",
    "experiment setup": "Experimental Setup",
    "experiments": "Experiments",
    "results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
    "bibliography": "References",
}


def normalize_heading(heading):
    """
    Convert recognized heading variants into canonical names.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        heading.strip().lower()
    )

    return SECTION_PATTERNS.get(normalized)


def identify_sections(text):
    """
    Identify common research paper sections.

    Returns:
        Dictionary mapping canonical section names to their text.

    Notes:
        - Section names use consistent capitalization.
        - Repeated sections with the same canonical name are merged.
        - Headings are detected at the beginning of a line.
    """

    if not isinstance(text, str) or not text.strip():
        return {}

    heading_names = sorted(
        SECTION_PATTERNS.keys(),
        key=len,
        reverse=True
    )

    alternatives = "|".join(
        re.escape(name)
        for name in heading_names
    )

    # Match headings at the beginning of a line.
    # Supports headings such as:
    # 1 Introduction
    # 2.1 Related Work
    # IV. Methodology
    # Introduction
    heading_pattern = re.compile(
        rf"""
        ^[ \t]*
        (?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]?[ \t]+)?
        (?P<heading>{alternatives})
        [ \t]*(?:[:.\-][ \t]*)?
        $
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE
    )

    matches = list(heading_pattern.finditer(text))

    sections = {}

    for index, match in enumerate(matches):
        raw_heading = match.group("heading")
        canonical_heading = normalize_heading(raw_heading)

        if canonical_heading is None:
            continue

        # Exclude the heading line from the section body.
        body_start = match.end()

        if index + 1 < len(matches):
            body_end = matches[index + 1].start()
        else:
            body_end = len(text)

        body = text[body_start:body_end].strip()

        if not body:
            continue

        if canonical_heading in sections:
            sections[canonical_heading] += "\n\n" + body
        else:
            sections[canonical_heading] = body

    return sections
