import re


def identify_sections(text):
    """
    Identify common research paper sections.
    """

    section_patterns = [
        "Abstract",
        "Introduction",
        "Related Work",
        "Background",
        "Methodology",
        "Methods",
        "Experimental Setup",
        "Experiments",
        "Results",
        "Discussion",
        "Conclusion",
        "References"
    ]

    sections = {}

    pattern = "|".join(
        re.escape(section)
        for section in section_patterns
    )

    matches = list(
        re.finditer(
            pattern,
            text,
            re.IGNORECASE
        )
    )

    for i, match in enumerate(matches):

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        heading = match.group()

        sections[heading] = text[
            start:end
        ].strip()

    return sections