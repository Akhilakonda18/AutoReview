"""
AutoReview: Online Publication Template Compliance Checker

Checks measurable PDF properties against requirements extracted from supplied
conference/journal guidelines. This is a prototype, not a guarantee of acceptance.

Supported inputs:
- Submitted paper PDF
- Optional official guidelines URL
- Optional local guidelines file: PDF, DOCX, TXT, MD, HTML

Important:
- YES means only that the supported checks passed.
- MISMATCH_DETECTED means at least one explicitly stated, measurable requirement failed.
- NOT_VERIFIED means there was not enough reliable information to decide.
- Some checks (e.g. anonymity, page numbers, margins, font identification) are
  heuristic and should be manually confirmed.
"""

from __future__ import annotations

import re
from pathlib import Path
from statistics import median
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # Compatibility with installations exposing the older module name


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    )
}
TIMEOUT = 20


def normalize(text: str) -> str:
    """Normalize whitespace and common PDF text artifacts."""
    text = (text or "").replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def read_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF."""
    chunks = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            chunks.append(page.get_text("text"))
    return "\n".join(chunks)


def read_guidelines_file(path: str | Path) -> str:
    """Read local guidelines from PDF, DOCX, TXT, MD, or HTML."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Guidelines file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text(file_path)

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "Reading DOCX guidelines requires python-docx. "
                "Install it with: pip install python-docx"
            ) from exc
        doc = Document(str(file_path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix in {".html", ".htm"}:
        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
        return soup.get_text(" ", strip=True)

    raise ValueError(
        f"Unsupported guidelines file type: {suffix}. "
        "Use PDF, DOCX, TXT, MD, HTML, or HTM."
    )


def fetch_url_text(url: str) -> str:
    """Fetch readable text from an HTML or PDF guidelines URL."""
    if not url:
        return ""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("The guidelines URL must start with http:// or https://")

    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    if "pdf" in content_type or parsed.path.lower().endswith(".pdf"):
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)

    if "html" in content_type or not content_type:
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
        return soup.get_text(" ", strip=True)

    return ""


def search_online(query: str, max_results: int = 5) -> list[dict]:
    """Return candidate links only; search results are not compliance evidence."""
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result"):
            link = item.select_one(".result__a")
            if not link:
                continue
            href = link.get("href", "")
            title = link.get_text(" ", strip=True)
            snippet_node = item.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            if href.startswith("http"):
                results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results
    except requests.RequestException:
        return []


def _get_font_records(page) -> list[dict]:
    records = []
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                size = span.get("size")
                font = span.get("font") or ""
                bbox = span.get("bbox")
                if text and isinstance(size, (int, float)):
                    records.append({
                        "text": text,
                        "size": float(size),
                        "font": font,
                        "bbox": tuple(bbox) if bbox else None,
                    })
    return records


def _page_size_label(width: float, height: float) -> str:
    short, long = sorted((width, height))
    if abs(short - 595.3) < 8 and abs(long - 841.9) < 8:
        return "A4"
    if abs(short - 612) < 8 and abs(long - 792) < 8:
        return "US Letter"
    return f"Other ({short:.0f} x {long:.0f} pt)"


def _estimate_columns(page, width: float) -> bool:
    """Heuristic: text blocks in both left and right regions suggest two columns."""
    blocks = [
        b for b in page.get_text("blocks")
        if len(b) >= 5 and str(b[4]).strip()
    ]
    left = right = 0
    for block in blocks:
        center_x = (block[0] + block[2]) / 2
        if center_x < width * 0.46:
            left += 1
        elif center_x > width * 0.54:
            right += 1
    return left >= 3 and right >= 3


def extract_pdf_information(pdf_path: str | Path) -> dict:
    """Extract text and measurable layout properties from a submitted PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: '{path}'")

    page_texts = []
    page_sizes = []
    page_records = []
    two_column_pages = 0
    all_font_records = []
    margin_estimates = []
    page_number_hits = 0

    with fitz.open(str(path)) as doc:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_index, page in enumerate(doc):
            page_texts.append(page.get_text("text"))
            width, height = float(page.rect.width), float(page.rect.height)
            page_sizes.append(_page_size_label(width, height))
            if _estimate_columns(page, width):
                two_column_pages += 1

            records = _get_font_records(page)
            all_font_records.extend(records)
            page_records.append(records)

            # Estimate text margins from text span bounding boxes. This is only
            # a heuristic; headers, footers, figures and equations affect it.
            boxes = [r["bbox"] for r in records if r["bbox"]]
            if boxes:
                left = min(b[0] for b in boxes)
                top = min(b[1] for b in boxes)
                right = width - max(b[2] for b in boxes)
                bottom = height - max(b[3] for b in boxes)
                margin_estimates.append({
                    "page": page_index + 1,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                })

            # Heuristic page-number detection in top/bottom 12% of page.
            footer_top = height * 0.88
            header_bottom = height * 0.12
            page_num_pattern = re.compile(
                rf"^\s*(?:page\s*)?{page_index + 1}(?:\s*(?:of|/)\s*\d+)?\s*$",
                re.IGNORECASE,
            )
            found = False
            for rec in records:
                if not rec["bbox"]:
                    continue
                y0, y1 = rec["bbox"][1], rec["bbox"][3]
                if (y0 <= header_bottom or y1 >= footer_top) and page_num_pattern.match(rec["text"]):
                    found = True
                    break
            if found:
                page_number_hits += 1

    text = "\n".join(page_texts)
    normalized = normalize(text)

    abstract_match = re.search(
        r"\babstract\b\s*[:—–-]?\s*(.*?)(?=\bkeywords?\b|\bindex terms\b|\bintroduction\b)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    abstract = abstract_match.group(1).strip() if abstract_match else ""

    section_patterns = {
        "Abstract": r"(?im)^\s*abstract\s*[:—–-]?\s*$",
        "Introduction": r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?introduction\s*$",
        "Methods": r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:methodology|methods|proposed method|proposed approach|materials and methods)\s*$",
        "Results": r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:results|experimental results|results and discussion|experiments and results)\s*$",
        "Conclusion": r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?conclusions?\s*$",
        "References": r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography)\s*$",
    }
    detected_sections = [
        name for name, pattern in section_patterns.items() if re.search(pattern, text)
    ]

    # Estimate the main-body page count from the earliest recognizable
    # end-matter heading. Impact Statement and Acknowledgements can appear
    # before References in some published PDFs. A page containing body text
    # (including a Limitations section) before one of these headings still
    # counts as a main-body page; end matter after the heading does not.
    # This remains a heuristic and the detected heading is reported for review.
    main_body_page_count = None
    references_boundary = None
    boundary_patterns = [
        ("references", re.compile(
            r"(?im)^\\s*(?:\\d+(?:\\.\\d+)*\\.?\\s*)?(?:references|bibliography)(?:\\s*[:—–-])?\\s*$"
        )),
        ("appendix/supplementary material", re.compile(
            r"(?im)^\\s*(?:appendix(?:\\s+[a-z0-9]+)?|appendices|supplementary material|supplemental material)\\s*$"
        )),
        ("impact statement", re.compile(
            r"(?im)^\\s*(?:impact statement|broader impact statement)\\s*$"
        )),
        ("acknowledgements", re.compile(
            r"(?im)^\\s*(?:acknowledgements|acknowledgments)\\s*$"
        )),
    ]
    for page_index, page_text in enumerate(page_texts):
        matches = []
        for label, pattern in boundary_patterns:
            match = pattern.search(page_text)
            if match:
                matches.append((match.start(), label, match))
        if not matches:
            continue

        _, boundary_label, match = min(matches, key=lambda item: item[0])
        text_before_heading = re.sub(r"\\s+", " ", page_text[:match.start()]).strip()

        # Count this page as main-body content only if there is substantive
        # text before the boundary. In particular, a Limitations section that
        # precedes an Impact Statement is part of the main paper, not end matter.
        same_page_has_body = len(text_before_heading) >= 120
        estimated_pages = page_index + (1 if same_page_has_body else 0)
        if page_index == 0 and same_page_has_body:
            estimated_pages = 1
        main_body_page_count = estimated_pages if estimated_pages > 0 else None
        references_boundary = {
            "page": page_index + 1,
            "heading": match.group(0).strip(),
            "type": boundary_label,
            "text_before_heading_characters": len(text_before_heading),
            "boundary_starts_mid_page": same_page_has_body,
        }
        break


    # Common identity and anonymization signals. These do not prove anonymity.
    identity_patterns = [
        r"\b(?:author|authors)\s*:",
        r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
        r"\b(university|department|institute|laboratory|company)\b",
        r"\b(corresponding author|author affiliation)\b",
    ]
    identity_matches = []
    for pattern in identity_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            identity_matches.append(pattern)

    # Most frequent font sizes and family names among extracted text spans.
    font_sizes = [round(r["size"], 1) for r in all_font_records]
    font_size_counts = {}
    for size in font_sizes:
        font_size_counts[size] = font_size_counts.get(size, 0) + 1
    common_font_sizes = sorted(
        font_size_counts.items(), key=lambda item: item[1], reverse=True
    )[:8]
    font_names = [r["font"] for r in all_font_records if r["font"]]
    font_name_counts = {}
    for name in font_names:
        font_name_counts[name] = font_name_counts.get(name, 0) + 1
    common_font_names = sorted(
        font_name_counts.items(), key=lambda item: item[1], reverse=True
    )[:8]

    return {
        "text": text,
        "normalized_text": normalized,
        "page_count": page_count,
        "main_body_page_count_estimate": main_body_page_count,
        "references_boundary": references_boundary,
        "page_sizes": page_sizes,
        "two_column_ratio": two_column_pages / page_count if page_count else 0.0,
        "abstract_word_count": len(re.findall(r"\b[\w'-]+\b", abstract)),
        "abstract_found": bool(abstract),
        "sections": detected_sections,
        "common_font_sizes_pt": common_font_sizes,
        "common_font_names": common_font_names,
        "margin_estimates_pt": margin_estimates,
        "page_number_hits": page_number_hits,
        "identity_signals_found": identity_matches,
    }


def find_page_limit(text: str) -> int | None:
    """Extract an explicit maximum total-page limit conservatively."""
    t = normalize(text)
    patterns = [
        r"\bmaximum\s+(?:of\s+)?(\d{1,2})\s+pages?\b",
        r"\blimited\s+to\s+(\d{1,2})\s+pages?\b",
        r"\bup\s+to\s+(\d{1,2})\s+pages?\b",
        r"\bnot\s+exceed\s+(\d{1,2})\s+pages?\b",
        r"\bshall\s+not\s+exceed\s+(\d{1,2})\s+pages?\b",
        r"\bno more than\s+(\d{1,2})\s+pages?\b",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, t, re.IGNORECASE):
            value = int(match.group(1))
            if 1 <= value <= 100:
                values.append(value)
    return min(values) if values else None


def guidelines_require_two_columns(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(
        r"\b(two|double)[ -]column\b|\btwo columns\b|\bdouble column format\b", t
    ))


def guidelines_require_letter(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(
        r"\b(us letter|letter size|8\.5\s*[x×]\s*11\s*(?:in|inches)?)\b", t
    ))


def guidelines_require_a4(text: str) -> bool:
    t = normalize(text)
    if re.search(r"\b(?:not|never|instead of|rather than)\s+(?:use\s+)?a4\b", t):
        return False
    if guidelines_require_letter(t):
        return False
    return bool(re.search(r"\b(a4|210\s*[x×]\s*297\s*mm)\b", t))


def is_generic_ieee_template_resource(url: str | None) -> bool:
    """Identify IEEE's general template-resource page, not venue-specific rules."""
    if not url:
        return False
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return (
        host.endswith("ieeeauthorcenter.ieee.org")
        and "authoring-tools-and-templates" in path
        and not any(token in path for token in ("conference-specific", "publication-guidelines"))
    )


def _extract_required_font_size(text: str) -> float | None:
    """Find a stated body font size, e.g. '10-point font' or 'font size of 10 pt'."""
    t = normalize(text)
    patterns = [
        r"\b(?:body\s+)?font(?:\s+size)?\s*(?:of|is|:)?\s*(\d{1,2}(?:\.\d+)?)\s*(?:-?\s*point|pt)\b",
        r"\b(\d{1,2}(?:\.\d+)?)\s*(?:-?\s*point|pt)\s+(?:body\s+)?font\b",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, t):
            value = float(match.group(1))
            if 6 <= value <= 16:
                values.append(value)
    return values[0] if values else None


def _guidelines_require_times(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(r"\b(times new roman|times font|times roman)\b", t))


def _extract_margin_requirement_inches(text: str) -> float | None:
    """Extract explicit all-side margin statements, e.g. '1 inch margins'."""
    t = normalize(text)
    patterns = [
        r"\b(?:all\s+)?margins?\s*(?:must be|should be|of|:)?\s*(\d(?:\.\d+)?)\s*(?:inches|inch|in\.)\b",
        r"\b(\d(?:\.\d+)?)\s*(?:inches|inch|in\.)\s+margins?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            value = float(match.group(1))
            if 0.2 <= value <= 2.5:
                return value
    return None


def _requirements_mention_page_numbers(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(r"\b(page numbers?|numbered pages?|number each page)\b", t))


def _requirements_demand_anonymity(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(
        r"\b(double[- ]blind|double anonym(?:ous|ity)|anonymous submission|"
        r"anonymized submission|remove author names|omit author names)\b", t
    ))


def _requirements_demand_banner(text: str) -> bool:
    t = normalize(text)
    return bool(re.search(r"\b(submission banner|banner on each page|conference banner)\b", t))


def _looks_like_page_numbering_is_present(paper: dict) -> bool:
    # A conservative heuristic: all or almost all pages have a detectable
    # sequential page number in the top/bottom margin.
    total = paper["page_count"]
    return total > 0 and paper["page_number_hits"] >= max(1, total - 1)


def _font_family_looks_like_times(paper: dict) -> bool:
    names = " ".join(name.lower() for name, _ in paper["common_font_names"])
    return "times" in names or "NimbusRoman".lower() in names or "liberationserif" in names


def _format_check(checks: list, reasons: list, name: str, passed, observed=None, note=None):
    item = {"check": name, "passed": passed}
    if observed is not None:
        item["observed"] = observed
    if note:
        item["note"] = note
    checks.append(item)
    if passed is False:
        reasons.append(f"Failed requirement: {name}.")


def check_paper_compliance(
    pdf_path: str | Path,
    venue_name: str,
    official_url: str | None = None,
    guidelines_path: str | Path | None = None,
    submission_stage: str = "anonymous submission",
) -> dict:
    """
    Compare a submitted paper with the supplied guidelines.

    Backward-compatible call:
        check_paper_compliance(pdf_path, venue_name, official_url=None)

    Optional local guideline file:
        check_paper_compliance(pdf_path, venue_name, official_url=None,
                               guidelines_path=None)

    submission_stage:
        "anonymous submission" (default), "camera-ready", or "published".
        Anonymity is enforced only for anonymous-submission stage.
    """
    stage = normalize(submission_stage).replace("_", "-")
    if stage in {"camera ready", "final version", "final", "camera-ready version"}:
        stage = "camera-ready"
    elif stage in {"published paper", "publication", "already published"}:
        stage = "published"
    elif stage in {"anonymous", "submission", "initial submission", "review submission", "double-blind submission"}:
        stage = "anonymous submission"
    elif stage not in {"anonymous submission", "camera-ready", "published"}:
        stage = "anonymous submission"

    try:
        paper = extract_pdf_information(pdf_path)
    except Exception as exc:
        return {
            "decision": "NO",
            "status": "PDF_READ_ERROR",
            "venue": venue_name,
            "reasons": [f"Could not read the submitted PDF: {exc}"],
            "checks": [],
            "sources": [],
        }

    guideline_parts = []
    sources = []
    errors = []

    if guidelines_path:
        try:
            guideline_text = read_guidelines_file(guidelines_path)
            if guideline_text.strip():
                guideline_parts.append(guideline_text)
                sources.append({
                    "title": Path(guidelines_path).name,
                    "url": "",
                    "source_type": "user-provided guidelines file",
                })
            else:
                errors.append("The supplied guidelines file contained no extractable text.")
        except Exception as exc:
            errors.append(f"Could not read the guidelines file: {exc}")

    if official_url:
        try:
            url_text = fetch_url_text(official_url)
            if url_text.strip():
                guideline_parts.append(url_text)
                sources.append({
                    "title": venue_name,
                    "url": official_url,
                    "source_type": "user-provided URL; official status not independently confirmed",
                })
            else:
                errors.append("No readable guideline text was extracted from the supplied URL.")
        except Exception as exc:
            errors.append(f"Could not retrieve the supplied guidelines URL: {exc}")

    if not guideline_parts and not official_url and not guidelines_path:
        candidates = search_online(f'"{venue_name}" official author guidelines template')
        return {
            "decision": "NOT VERIFIED",
            "status": "NOT_VERIFIED",
            "venue": venue_name,
            "reasons": [
                "No readable guidelines were provided or retrieved.",
                "Search results alone are not reliable enough to decide compliance.",
                *errors,
            ],
            "checks": [],
            "sources": candidates,
        }

    guideline_text = "\n".join(guideline_parts)
    normalized_guidelines = normalize(guideline_text)
    generic_ieee_resource = is_generic_ieee_template_resource(official_url)

    if not normalized_guidelines:
        return {
            "decision": "NOT VERIFIED",
            "status": "NOT_VERIFIED",
            "venue": venue_name,
            "reasons": [
                "The supplied guideline source could not be read as text.",
                "Try a text-based PDF/DOCX/TXT file or the venue's official HTML/PDF guidelines URL.",
                *errors,
            ],
            "checks": [],
            "sources": sources,
        }

    checks = []
    reasons = []

    if generic_ieee_resource:
        checks.append({
            "check": "Venue-specific IEEE requirements",
            "passed": None,
            "note": (
                "This is IEEE's general template-resource page, not a specific conference's "
                "author guidelines. Venue-specific limits and formatting requirements cannot "
                "be inferred from this page alone."
            ),
        })

    # Page limit: special handling for a main-body limit that explicitly excludes references.
    is_icml_2025 = "icml" in venue_name.lower() and "2025" in venue_name
    if is_icml_2025:
        page_limit = 8
        body_pages = paper["main_body_page_count_estimate"]
        if body_pages is None:
            _format_check(
                checks, reasons, "ICML 2025 main-body page limit (maximum 8 pages)",
                None,
                observed=f"{paper['page_count']} total PDF pages; main-body boundary not detected",
                note="References/appendix boundary could not be confidently detected; check manually.",
            )
        else:
            passed = body_pages <= page_limit
            _format_check(
                checks, reasons, "ICML 2025 main-body page limit (maximum 8 pages)",
                passed,
                observed={
                    "estimated_main_body_pages": body_pages,
                    "total_pdf_pages": paper["page_count"],
                    "detected_boundary": paper.get("references_boundary"),
                },
                note=("Estimated from the earliest recognizable References/Bibliography/Appendix heading. "
                      "If the heading is unusual or the body continues around it, verify manually."),
            )
    else:
        page_limit = None if generic_ieee_resource else find_page_limit(guideline_text)
        if page_limit is not None:
            passed = paper["page_count"] <= page_limit
            _format_check(
                checks, reasons, f"Page limit (maximum {page_limit} pages)",
                passed, observed=paper["page_count"],
                note="This checks total PDF pages. If references are excluded by the rule, verify the boundary manually.",
            )
        else:
            _format_check(
                checks, reasons, "Page limit", None,
                observed=paper["page_count"],
                note="No clear maximum total-page limit could be extracted from the guidelines.",
            )

    # Column layout.
    if not generic_ieee_resource and guidelines_require_two_columns(normalized_guidelines):
        passed = paper["two_column_ratio"] >= 0.6
        _format_check(
            checks, reasons, "Two-column layout", passed,
            observed=f"{paper['two_column_ratio']:.0%} of pages appear two-column",
            note="Column detection is heuristic and can be affected by figures, tables, and title pages.",
        )
    else:
        _format_check(
            checks, reasons, "Column layout", None,
            note="The supplied text did not establish an applicable two-column requirement.",
        )

    # Page size.
    if generic_ieee_resource:
        _format_check(
            checks, reasons, "Page size", None, observed=paper["page_sizes"],
            note="The general IEEE template page links to multiple templates and does not establish one page size for every conference.",
        )
    elif is_icml_2025 or guidelines_require_letter(normalized_guidelines):
        bad_pages = [i + 1 for i, size in enumerate(paper["page_sizes"]) if size != "US Letter"]
        _format_check(
            checks, reasons, "US Letter page size", not bad_pages,
            observed=paper["page_sizes"],
        )
        if bad_pages:
            reasons.append(f"Page(s) {bad_pages} do not appear to be US Letter size.")
    elif guidelines_require_a4(normalized_guidelines):
        bad_pages = [i + 1 for i, size in enumerate(paper["page_sizes"]) if size != "A4"]
        _format_check(checks, reasons, "A4 page size", not bad_pages, observed=paper["page_sizes"])
        if bad_pages:
            reasons.append(f"Page(s) {bad_pages} do not appear to be A4 size.")
    else:
        _format_check(
            checks, reasons, "Page size", None, observed=paper["page_sizes"],
            note="The guidelines did not clearly specify A4 or US Letter size.",
        )

    # Body font size: only evaluate if guidelines clearly state a size.
    required_font_size = _extract_required_font_size(guideline_text)
    if required_font_size is not None:
        sizes = paper["common_font_sizes_pt"]
        # Ignore very rare sizes (e.g. symbols) by evaluating the most common span size.
        observed_size = sizes[0][0] if sizes else None
        passed = observed_size is not None and abs(observed_size - required_font_size) <= 0.6
        _format_check(
            checks, reasons, f"Common text font size ({required_font_size:g} pt required)",
            passed,
            observed=f"most frequent extracted size: {observed_size} pt; common sizes: {sizes}",
            note="This is a heuristic across all text spans; titles, captions, footnotes, and references may legitimately use other sizes.",
        )
    else:
        _format_check(
            checks, reasons, "Font size", None,
            observed=paper["common_font_sizes_pt"],
            note="No explicit body-font size could be extracted from the guidelines.",
        )

    # Font family.
    if _guidelines_require_times(normalized_guidelines):
        passed = _font_family_looks_like_times(paper)
        _format_check(
            checks, reasons, "Times-style font family",
            passed,
            observed=paper["common_font_names"],
            note="PDF font names can differ from the requested family after embedding or substitution; confirm manually.",
        )
    else:
        _format_check(
            checks, reasons, "Font family", None,
            observed=paper["common_font_names"],
            note="No specific font family was clearly required in the extracted guideline text.",
        )

    # Margins: compare conservative minimum observed margins with explicit requirement.
    margin_inches = _extract_margin_requirement_inches(guideline_text)
    if margin_inches is not None and paper["margin_estimates_pt"]:
        required_pt = margin_inches * 72
        # A small tolerance avoids rejecting due to minor PDF measurement differences.
        failed_pages = []
        for m in paper["margin_estimates_pt"]:
            if min(m["left"], m["top"], m["right"], m["bottom"]) < required_pt - 5:
                failed_pages.append(m["page"])
        _format_check(
            checks, reasons, f"Margins (approximately {margin_inches:g} inch minimum)",
            not failed_pages,
            observed=f"estimated margins in points for {len(paper['margin_estimates_pt'])} pages",
            note="Text bounding boxes are only an approximation of page margins; figures and headers can affect the estimate.",
        )
        if failed_pages:
            reasons.append(f"Estimated text margins may be below the requirement on page(s) {failed_pages}.")
    else:
        _format_check(
            checks, reasons, "Margins", None,
            observed=f"{len(paper['margin_estimates_pt'])} pages with estimates",
            note="No explicit margin requirement was extracted, or margin measurements were unavailable.",
        )

    # Page numbers: check only when guidelines explicitly require them.
    if _requirements_mention_page_numbers(normalized_guidelines):
        passed = _looks_like_page_numbering_is_present(paper)
        _format_check(
            checks, reasons, "Page numbering",
            passed,
            observed=f"detectable page numbers on {paper['page_number_hits']} of {paper['page_count']} pages",
            note="Detection is heuristic; roman numerals, footer graphics, and custom numbering may not be recognized.",
        )
    else:
        _format_check(checks, reasons, "Page numbering", None,
                      note="The guidelines did not clearly require page numbers.")

    # Anonymity depends on the submission stage. Conference guidelines often
    # describe double-blind review and camera-ready publication on the same page.
    if stage == "anonymous submission" and _requirements_demand_anonymity(normalized_guidelines):
        identity_found = bool(paper["identity_signals_found"])
        _format_check(
            checks, reasons, "Preliminary anonymization check",
            not identity_found,
            observed="possible author/affiliation/email signals found" if identity_found else "no common identity signals detected",
            note="This is a preliminary heuristic, not proof of anonymity. Manually inspect acknowledgements, self-citations, PDF metadata, and linked artifacts.",
        )
    elif stage in {"camera-ready", "published"}:
        _format_check(
            checks, reasons, "Anonymization",
            None,
            observed=f"paper stage: {stage}",
            note="Author names and affiliations are normally expected in camera-ready/published papers, so identity signals were not treated as a failure. Other applicable requirements are still checked.",
        )
    else:
        _format_check(checks, reasons, "Anonymization", None,
                      observed=f"paper stage: {stage}",
                      note="The supplied guidelines did not clearly require anonymous submission, or the anonymity requirement could not be established.")

    # Submission banner: only inspect if the extracted rules mention one.
    if _requirements_demand_banner(normalized_guidelines):
        has_banner = bool(re.search(r"\b(submission|conference|paper)\s+(?:id|number|banner)\b", paper["normalized_text"]))
        _format_check(
            checks, reasons, "Submission banner",
            has_banner,
            observed="possible banner text detected" if has_banner else "no obvious banner text detected",
            note="Banner wording varies; visual placement and exact wording require manual verification.",
        )
    else:
        _format_check(checks, reasons, "Submission banner", None,
                      note="No explicit submission-banner requirement was extracted.")

    # Common section headings are observations, not mandatory requirements.
    checks.append({
        "check": "Common section headings detected (informational only)",
        "passed": None,
        "observed": paper["sections"],
        "note": "Section names vary by venue. Missing a generic heading is not treated as a failure unless explicitly required.",
    })

    # Abstract word count is checked only when a clear maximum is found.
    abstract_limit_match = re.search(
        r"\babstract\b.{0,100}?\b(?:maximum|no more than|up to|limited to)\s+(\d{2,3})\s+words?\b",
        normalized_guidelines,
    )
    if abstract_limit_match:
        abstract_limit = int(abstract_limit_match.group(1))
        passed = paper["abstract_word_count"] <= abstract_limit
        _format_check(
            checks, reasons, f"Abstract length (maximum {abstract_limit} words)",
            passed, observed=paper["abstract_word_count"],
        )
    else:
        _format_check(
            checks, reasons, "Abstract length", None,
            observed=f"{paper['abstract_word_count']} words (estimated)",
            note="No clear maximum abstract word count was extracted.",
        )

    hard_failures = [c for c in checks if c.get("passed") is False]
    tested_checks = [c for c in checks if c.get("passed") is not None]

    if hard_failures:
        decision = "NO"
        status = "MISMATCH_DETECTED"
        reasons.insert(0, f"{len(hard_failures)} supported compliance check(s) failed.")
    elif not tested_checks:
        decision = "NOT VERIFIED"
        status = "NOT_VERIFIED"
        reasons.append(
            "No applicable formatting requirement could be verified automatically. "
            "This is not a rejection: provide the venue's specific author guidelines or exact template."
        )
    else:
        decision = "YES"
        status = "LIMITED_CHECKS_PASSED"
        reasons.append(
            "No mismatch was found in the limited checks that could be tested. "
            "This does not prove exact template identity or guarantee acceptance."
        )

    if errors:
        reasons.extend(errors)

    return {
        "decision": decision,
        "status": status,
        "venue": venue_name,
        "page_count": paper["page_count"],
        "page_sizes": paper["page_sizes"],
        "two_column_ratio": round(paper["two_column_ratio"], 3),
        "abstract_word_count": paper["abstract_word_count"],
        "sections": paper["sections"],
        "common_font_sizes_pt": paper["common_font_sizes_pt"],
        "common_font_names": paper["common_font_names"],
        "margin_estimates_pt": paper["margin_estimates_pt"],
        "page_number_hits": paper["page_number_hits"],
        "checks": checks,
        "reasons": reasons,
        "sources": sources,
    }


if __name__ == "__main__":
    print("AutoReview Online Template Compliance Checker")
    print("---------------------------------------------")
    pdf_path = input("Path to submitted paper PDF: ").strip().strip('"')
    venue_name = input("Exact conference/journal name: ").strip()
    guidelines_path = input(
        "Path to guidelines PDF/DOCX/TXT/HTML (press Enter to skip): "
    ).strip().strip('"')
    official_url = input(
        "Official guidelines URL (press Enter to skip): "
    ).strip()
    submission_stage = input(
        "Paper stage [anonymous submission / camera-ready / published] (default: anonymous submission): "
    ).strip() or "anonymous submission"

    result = check_paper_compliance(
        pdf_path=pdf_path,
        venue_name=venue_name,
        official_url=official_url or None,
        guidelines_path=guidelines_path or None,
        submission_stage=submission_stage,
    )

    print("\nDECISION:", result["decision"])
    print("STATUS:", result["status"])
    print("TARGET:", result["venue"])
    print("\nCHECKS:")
    for check in result.get("checks", []):
        passed = check.get("passed")
        label = "PASS" if passed is True else "FAIL" if passed is False else "INFO"
        print(f"- {label}: {check['check']}")
        if "observed" in check:
            print("  Observed:", check["observed"])
        if check.get("note"):
            print("  Note:", check["note"])

    print("\nREASONS:")
    for reason in result.get("reasons", []):
        print("-", reason)

    print("\nSOURCES:")
    for source in result.get("sources", []):
        print("-", source.get("title", "Guidelines source"))
        if source.get("url"):
            print(" ", source["url"])
