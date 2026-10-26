
from __future__ import annotations

import json
import re
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

# ============================================================
# AUTOREVIEW CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_DIRS = [
    PROJECT_DIR / "data" / "reference papers",
    PROJECT_DIR / "data" / "reference_papers",
]

HISTORY_FILE = OUTPUT_DIR / "review_history.json"

st.set_page_config(
    page_title="AutoReview | Research Paper Evaluation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
    color: #111827;
}

/* Main content layout */
[data-testid="stMain"] {
    background-color: #f5f7fb;
}

[data-testid="stMain"] .block-container {
    max-width: 1400px;
    padding: 2rem 2rem 3rem 2rem;
}

/* Main headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #111827 !important;
    font-weight: 700;
}

/* Main paragraph text */
[data-testid="stMain"] p {
    color: #475569;
    line-height: 1.7;
}

/* Form labels */
[data-testid="stMain"] label,
[data-testid="stMain"] .stMarkdown {
    color: #111827;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] > div {
    background-color: #111827;
}

/* Sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #f9fafb;
}

/* Sidebar navigation labels */
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 5px;
    color: #f9fafb !important;
}

[data-testid="stSidebar"] .stRadio label p {
    color: #f9fafb !important;
}

/* Brand */
.brand {
    font-size: 29px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #ffffff;
    margin-bottom: 2px;
}

.brand-accent {
    color: #60a5fa;
}

.sidebar-subtitle {
    color: #cbd5e1 !important;
    font-size: 12px;
    margin-bottom: 25px;
}

/* Page hero */
.hero {
    padding: 30px;
    border-radius: 18px;
    color: #111827;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    margin-bottom: 22px;
}

.hero h1 {
    color: #111827 !important;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #475569 !important;
    font-size: 15px;
    line-height: 1.7;
}

/* Section headings */
.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #111827;
    margin-top: 16px;
    margin-bottom: 12px;
}

/* Dashboard metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #64748b;
    font-size: 12px;
    margin-top: 4px;
}

/* Feature cards */
.feature-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 19px;
    min-height: 145px;
    margin-bottom: 10px;
}

.feature-card h3 {
    font-size: 17px;
    color: #111827 !important;
    margin-bottom: 8px;
}

.feature-card p {
    font-size: 13px;
    color: #64748b !important;
}

/* Status labels */
.status-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 30px;
    background: #dbeafe;
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 700;
}

/* Information boxes */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 13px 16px;
    border-radius: 7px;
    color: #1e3a8a;
    margin: 12px 0;
}

/* Buttons */
div.stButton > button[kind="primary"] {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 9px;
    font-weight: 650;
    min-height: 43px;
}

div.stButton > button {
    border-radius: 9px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding: 25px 0 10px;
}
</style>

    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "review_result" not in st.session_state:
    st.session_state.review_result = None

if "plagiarism_result" not in st.session_state:
    st.session_state.plagiarism_result = None

if "template_result" not in st.session_state:
    st.session_state.template_result = None

if "uploaded_paper_path" not in st.session_state:
    st.session_state.uploaded_paper_path = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def read_history():
    """Read locally saved review history."""
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_history(record):
    """Save a compact review-history record."""
    history = read_history()
    history.insert(0, record)
    history = history[:100]

    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_uploaded_pdf(uploaded_file):
    """Save an uploaded PDF to a unique temporary project file."""
    upload_dir = OUTPUT_DIR / "uploaded_papers"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        Path(uploaded_file.name).name,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = upload_dir / f"{timestamp}_{safe_name}"

    destination.write_bytes(uploaded_file.getvalue())

    return destination


def format_score(value):
    """Display a score when it is available."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.1f}/10"
    except (TypeError, ValueError):
        return str(value)


def display_metric(label, value, note=""):
    """Render a dashboard metric."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_reference_directories():
    """Return existing local reference-paper directories."""
    return [str(p) for p in REFERENCE_DIRS if p.is_dir()]


def show_file_download(path, label):
    """Show a download button for an existing file."""
    file_path = Path(path)

    if not file_path.is_file():
        return

    try:
        data = file_path.read_bytes()

        st.download_button(
            label=label,
            data=data,
            file_name=file_path.name,
            mime=(
                "application/json"
                if file_path.suffix.lower() == ".json"
                else "text/plain"
            ),
            key=f"download_{file_path.name}",
        )
    except OSError as error:
        st.error(f"Unable to read the report: {error}")


def display_uploaded_paper_selector(key):
    """Provide a PDF upload control and save the selected paper."""
    uploaded = st.file_uploader(
        "Upload research paper (PDF)",
        type=["pdf"],
        key=key,
        help="Upload a text-based research paper PDF.",
    )

    if uploaded is None:
        return None

    if uploaded.size == 0:
        st.error("The uploaded file is empty.")
        return None

    try:
        saved_path = save_uploaded_pdf(uploaded)
        st.session_state.uploaded_paper_path = str(saved_path)

        st.success(f"Paper uploaded: {uploaded.name}")
        st.caption(f"File size: {uploaded.size / 1024:.1f} KB")

        return saved_path

    except OSError as error:
        st.error(f"Could not save the uploaded PDF: {error}")
        return None


# ============================================================
# BACKEND: FULL PAPER REVIEW
# ============================================================

def run_complete_review(pdf_path):
    """
    Run the existing main.py review pipeline against an uploaded PDF.

    The pipeline expects a PDF_PATH variable. We temporarily set it
    to the uploaded paper and restore the original value afterwards.
    """
    import importlib

    main_module = importlib.import_module("main")

    if not hasattr(main_module, "run_full_review"):
        raise AttributeError(
            "main.py does not expose run_full_review()."
        )

    original_pdf_path = getattr(
        main_module,
        "PDF_PATH",
        "data/sample papers/paper.pdf",
    )

    try:
        main_module.PDF_PATH = str(pdf_path)

        # The current pipeline prints output and returns its result.
        result = main_module.run_full_review()

    finally:
        main_module.PDF_PATH = original_pdf_path

    if not isinstance(result, dict):
        raise RuntimeError(
            "The review pipeline did not return a result. "
            "Check the terminal for the pipeline error."
        )

    return result


# ============================================================
# BACKEND: PLAGIARISM CHECK
# ============================================================

def run_plagiarism_analysis(pdf_path, online_arxiv=False, arxiv_query=""):
    """Run the existing plagiarism checker."""
    from plagiarism_checker import run_plagiarism_check

    reference_dirs = get_reference_directories()

    result = run_plagiarism_check(
        str(pdf_path),
        reference_dirs=reference_dirs or None,
        offline=not online_arxiv,
        arxiv_query=arxiv_query.strip() if online_arxiv else None,
        arxiv_results=5,
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "The plagiarism checker returned an unexpected result."
        )

    return result


# ============================================================
# BACKEND: TEMPLATE AND FORMATTING CHECK
# ============================================================

DEFAULT_TEMPLATES = {
    "General Research Paper": {
        "required_sections": [
            "Abstract",
            "Introduction",
            "Methodology",
            "Results",
            "Conclusion",
            "References",
        ],
        "minimum_pages": 3,
        "maximum_pages": 30,
        "expected_page_width_inches": 8.27,
        "expected_page_height_inches": 11.69,
        "page_size_tolerance_inches": 0.45,
    },
    "IEEE-Style Research Paper": {
        "required_sections": [
            "Abstract",
            "Index Terms",
            "Introduction",
            "Related Work",
            "Methodology",
            "Results",
            "Conclusion",
            "References",
        ],
        "minimum_pages": 3,
        "maximum_pages": 15,
        "expected_page_width_inches": 8.27,
        "expected_page_height_inches": 11.69,
        "page_size_tolerance_inches": 0.45,
    },
    "Conference Paper - Configurable": {
        "required_sections": [
            "Abstract",
            "Introduction",
            "Related Work",
            "Methodology",
            "Experiments",
            "Conclusion",
            "References",
        ],
        "minimum_pages": 2,
        "maximum_pages": 12,
        "expected_page_width_inches": 8.27,
        "expected_page_height_inches": 11.69,
        "page_size_tolerance_inches": 0.45,
    },
}


SECTION_ALIASES = {
    "abstract": ["abstract", "summary"],
    "index terms": ["index terms", "keywords", "key words"],
    "introduction": ["introduction", "1 introduction", "1. introduction"],
    "related work": [
        "related work",
        "related works",
        "literature review",
        "literature survey",
        "background",
    ],
    "methodology": [
        "methodology",
        "method",
        "proposed method",
        "proposed methodology",
        "materials and methods",
    ],
    "results": [
        "results",
        "experimental results",
        "results and discussion",
        "evaluation",
    ],
    "experiments": [
        "experiments",
        "experimental setup",
        "experiments and results",
        "experimental evaluation",
    ],
    "conclusion": [
        "conclusion",
        "conclusions",
        "conclusion and future work",
        "conclusions and future work",
    ],
    "references": [
        "references",
        "bibliography",
        "works cited",
    ],
}


def detect_sections(text):
    """Find likely section headings in extracted PDF text."""
    detected = []

    for line in text.splitlines():
        heading = re.sub(r"\s+", " ", line.strip()).strip(" .:-")

        if not heading or len(heading) > 120:
            continue

        normalized = heading.lower()

        for section, aliases in SECTION_ALIASES.items():
            for alias in aliases:
                pattern = (
                    r"^(?:(?:\d+(?:\.\d+)*|[IVX]+)"
                    r"[\s.)-]+)?"
                    + re.escape(alias)
                    + r"$"
                )

                if re.match(pattern, normalized, re.IGNORECASE):
                    detected.append(section)
                    break

    return sorted(set(detected))


def run_template_check(
    pdf_path,
    template_name,
    custom_sections,
    min_pages,
    max_pages,
    check_page_size,
    submission_stage="Anonymous Submission",
):
    """
    Check required sections and basic PDF properties.

    This is a configurable screening tool, not a complete
    reproduction of an official journal or conference validator.
    """
    import pymupdf

    template = dict(DEFAULT_TEMPLATES.get(template_name, {}))

    required_sections = (
        list(custom_sections)
        if custom_sections
        else list(template.get("required_sections", []))
    )

    if not required_sections:
        required_sections = [
            "Abstract",
            "Introduction",
            "Methodology",
            "Results",
            "Conclusion",
            "References",
        ]

    checks = []

    with pymupdf.open(str(pdf_path)) as document:
        page_count = len(document)

        if page_count == 0:
            raise ValueError("The PDF contains no pages.")

        extracted_text = "\n".join(
            page.get_text("text") for page in document
        )

        if not extracted_text.strip():
            raise ValueError(
                "No text could be extracted from this PDF. "
                "The paper may be scanned and require OCR."
            )

        # Stage-specific screening. Anonymous submissions should not expose
        # identifying author details in the first-page author block. This is
        # deliberately a REVIEW flag, not an automatic rejection, because
        # PDF text heuristics can produce false positives.
        stage_normalized = submission_stage.strip().lower()
        if stage_normalized == "anonymous submission":
            first_page_text = document[0].get_text("text")[:5000]
            identifying_patterns = [
                r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
                r"(?i)\b(university|department|institute|laboratory|lab|company)\b",
                r"(?i)\b( author|authors|corresponding author)\b",
            ]
            import re
            matched_identifiers = [
                pattern for pattern in identifying_patterns
                if re.search(pattern, first_page_text)
            ]
            checks.append({
                "Check": "Anonymous-submission author details",
                "Result": "REVIEW" if matched_identifiers else "REVIEW",
                "Details": (
                    "Potential author-identifying information may be present "
                    "in the first page. Manually verify names, affiliations, "
                    "email addresses, acknowledgements, and document metadata."
                    if matched_identifiers else
                    "No obvious identifying pattern was detected by the basic "
                    "text scan. Manually verify the PDF and its metadata; this "
                    "scan cannot guarantee anonymity."
                ),
            })
        elif stage_normalized == "camera-ready":
            checks.append({
                "Check": "Camera-ready author information",
                "Result": "REVIEW",
                "Details": (
                    "Confirm that author names, affiliations, acknowledgements, "
                    "and final declarations follow the selected venue's "
                    "camera-ready instructions."
                ),
            })
        else:  # Published Paper
            checks.append({
                "Check": "Published-paper requirements",
                "Result": "REVIEW",
                "Details": (
                    "Confirm final publication elements required by the venue, "
                    "such as DOI, copyright notice, publisher formatting, and "
                    "the published version's metadata, where applicable."
                ),
            })

        detected = detect_sections(extracted_text)

        normalized_detected = set(detected)

        missing = []

        for required in required_sections:
            key = required.strip().lower()
            aliases = SECTION_ALIASES.get(key, [key])

            if not any(
                alias.lower() in normalized_detected
                for alias in aliases
            ):
                # Use a second text search to reduce false
                # missing-section warnings when formatting varies.
                pattern = (
                    r"(?im)^\s*(?:\d+(?:\.\d+)*[\s.)-]+)?"
                    + re.escape(required.strip())
                    + r"\s*[:.]?\s*$"
                )

                if not re.search(pattern, extracted_text):
                    missing.append(required)

        checks.append({
            "Check": "Required sections",
            "Result": (
                "PASS" if not missing else "FAIL"
            ),
            "Details": (
                "All configured sections were detected."
                if not missing
                else "Missing: " + ", ".join(missing)
            ),
        })

        checks.append({
            "Check": "Page count",
            "Result": (
                "PASS"
                if min_pages <= page_count <= max_pages
                else "FAIL"
            ),
            "Details": (
                f"{page_count} pages; configured range "
                f"{min_pages}-{max_pages}."
            ),
        })

        # Page-size check: use the first page as a screening
        # measurement; PDFs can have intentionally mixed sizes.
        first_page = document[0]
        width_in = first_page.rect.width / 72
        height_in = first_page.rect.height / 72

        if check_page_size:
            expected_w = float(
                template.get("expected_page_width_inches", 8.27)
            )
            expected_h = float(
                template.get("expected_page_height_inches", 11.69)
            )
            tolerance = float(
                template.get("page_size_tolerance_inches", 0.45)
            )

            is_portrait = (
                abs(width_in - expected_w) <= tolerance
                and abs(height_in - expected_h) <= tolerance
            )

            is_landscape = (
                abs(width_in - expected_h) <= tolerance
                and abs(height_in - expected_w) <= tolerance
            )

            checks.append({
                "Check": "Page dimensions",
                "Result": (
                    "PASS" if is_portrait or is_landscape else "REVIEW"
                ),
                "Details": (
                    f"First page: {width_in:.2f} × "
                    f"{height_in:.2f} inches. "
                    f"Configured reference: {expected_w:.2f} × "
                    f"{expected_h:.2f} inches."
                ),
            })

        # Report observed font sizes, not an automatic pass/fail.
        font_sizes = []

        for page in document:
            try:
                page_dict = page.get_text("dict")

                for block in page_dict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size")

                            if size and span.get("text", "").strip():
                                font_sizes.append(float(size))
            except Exception:
                continue

        if font_sizes:
            font_sizes.sort()

            median_size = font_sizes[len(font_sizes) // 2]

            checks.append({
                "Check": "Observed font size",
                "Result": "REVIEW",
                "Details": (
                    f"Median extracted text size: "
                    f"{median_size:.1f} pt. "
                    "Compare this with the selected official template."
                ),
            })
        else:
            checks.append({
                "Check": "Observed font size",
                "Result": "REVIEW",
                "Details": (
                    "Font-size information was not available."
                ),
            })

    failed = [
        check for check in checks
        if check["Result"] == "FAIL"
    ]

    needs_review = [
        check for check in checks
        if check["Result"] == "REVIEW"
    ]

    if failed:
        eligibility = "REJECTED / INCOMPLETE"
    elif needs_review:
        eligibility = "NEEDS MANUAL REVIEW"
    else:
        eligibility = "PASSES CONFIGURED CHECKS"

    return {
        "template": template_name,
        "submission_stage": submission_stage,
        "required_sections": required_sections,
        "detected_sections": detected,
        "missing_sections": missing,
        "page_count": page_count,
        "checks": checks,
        "eligibility": eligibility,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "This is an automated preliminary screening. "
            "It does not guarantee compliance with every official "
            "journal or conference formatting rule."
        ),
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">Auto<span class="brand-accent">Review</span></div>
        <div class="sidebar-subtitle">
            AI-powered research paper evaluation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    selected_page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📄 Complete Paper Review",
            "🔍 Plagiarism Check",
            "📋 Template Compliance",
            "⭐ Paper Score",
            "🛡️ Evidence Verification",
            "📊 Reports & History",
        ],
        key="navigation",
    )

    st.markdown("---")

    st.caption("PROJECT STATUS")
    st.markdown(
        '<span class="status-pill">Backend integration</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("AutoReview · Research Paper Assistant")


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if selected_page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Research smarter with AutoReview.</h1>
            <p>
                Evaluate research papers, identify potential similarities,
                check submission requirements, and review paper quality
                from one workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = read_history()

    completed_reviews = [
        item for item in history
        if item.get("type") == "Full Paper Review"
    ]

    latest_score = (
        completed_reviews[0].get("overall_score", "N/A")
        if completed_reviews else "N/A"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        display_metric(
            "Completed Reviews",
            len(completed_reviews),
            "Saved full paper reviews",
        )

    with col2:
        display_metric(
            "Latest Paper Score",
            format_score(latest_score)
            if latest_score != "N/A"
            else "N/A",
            "Most recent completed review",
        )

    with col3:
        display_metric(
            "Available Tools",
            "3",
            "Review and evaluation tools",
        )

    st.markdown(
        '<div class="section-title">Evaluation Workspace</div>',
        unsafe_allow_html=True,
    )

    main_tools = [
        (
            "📄 Complete Paper Review",
            "Run seven specialist agents and generate a consolidated review.",
        ),
        (
            "🔍 Plagiarism Check",
            "Compare submitted text against available local reference PDFs.",
        ),
        (
            "📋 Template Compliance",
            "Check mandatory sections, page count, and selected PDF properties.",
        ),
    ]

    cols = st.columns(3)

    for col, card in zip(cols, main_tools):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h3>{card[0]}</h3>
                    <p>{card[1]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Supporting features—not counted as separate review tools
    st.markdown(
        '<div class="section-title">Supporting Features</div>',
        unsafe_allow_html=True,
    )

    supporting_features = [
        (
            "⭐ Paper Score",
            "Review the overall score and individual specialist scores.",
        ),
        (
            "🛡️ Evidence Verification",
            "Inspect verification statuses and evidence gaps.",
        ),
        (
            "📊 Reports & History",
            "Access generated reports and locally saved review history.",
        ),
    ]

    cols = st.columns(3)

    for col, card in zip(cols, supporting_features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h3>{card[0]}</h3>
                    <p>{card[1]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


    st.info(
        "Choose a tool from the sidebar to upload your paper and "
        "start an evaluation. Full paper reviews can make multiple "
        "LLM requests and may consume API quota."
    )

    if completed_reviews:
        st.markdown("### Recent reviews")

        st.dataframe(
            [
                {
                    "Date": item.get("date", ""),
                    "Paper": item.get("paper", ""),
                    "Score": format_score(item.get("overall_score")),
                    "Status": item.get("status", "Completed"),
                }
                for item in completed_reviews[:5]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 2: COMPLETE PAPER REVIEW
# ============================================================

elif selected_page == "📄 Complete Paper Review":

    st.title("📄 Complete Paper Review")

    st.write(
        "Run the complete AutoReview pipeline using your seven "
        "specialist agents, evidence verification, score fusion, "
        "and final reviewer."
    )

    st.warning(
        "A full review may use multiple Gemini API requests. "
        "If you receive a quota error, wait before trying again."
    )

    uploaded_path = display_uploaded_paper_selector(
        "complete_review_upload"
    )

    if uploaded_path:
        st.markdown("#### Review configuration")

        run_plagiarism = st.checkbox(
            "Include the existing plagiarism check",
            value=True,
        )

        if st.button(
            "🚀 Run Complete Review",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Running the complete review pipeline. "
                    "Please keep this page open..."
                ):
                    # Respect the user's selection without changing
                    # the pipeline's default configuration permanently.
                    import importlib

                    main_module = importlib.import_module("main")

                    old_flag = getattr(
                        main_module,
                        "RUN_PLAGIARISM_CHECK",
                        True,
                    )

                    try:
                        main_module.RUN_PLAGIARISM_CHECK = run_plagiarism
                        result = run_complete_review(uploaded_path)
                    finally:
                        main_module.RUN_PLAGIARISM_CHECK = old_flag

                st.session_state.review_result = result

                report_path = result.get("report_path")

                save_history({
                    "type": "Full Paper Review",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "paper": uploaded_path.name,
                    "overall_score": result.get("overall_score"),
                    "status": "Completed",
                    "report_path": str(report_path or ""),
                })

                st.success("The complete review finished.")

            except Exception as error:
                st.session_state.last_error = str(error)

                st.error(
                    f"Could not complete the review: {error}"
                )

                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

    result = st.session_state.review_result

    if result:
        st.markdown("---")
        st.subheader("Review Results")

        score = result.get("overall_score", "N/A")

        c1, c2, c3 = st.columns(3)

        with c1:
            display_metric(
                "Overall Score",
                format_score(score),
                "Score returned by the existing score-fusion module",
            )

        with c2:
            display_metric(
                "Specialist Agents",
                len(result.get("agent_results", {})),
                "Generated specialist reviews",
            )

        with c3:
            verification = result.get("verification_results", {})
            display_metric(
                "Verified Reviews",
                sum(
                    1 for value in verification.values()
                    if value.get("status") == "SUPPORTED"
                ),
                "Marked SUPPORTED by the verifier",
            )

        scores = result.get("agent_scores", {})

        if scores:
            st.subheader("Individual Agent Scores")

            st.dataframe(
                [
                    {
                        "Agent": agent,
                        "Score": value,
                    }
                    for agent, value in scores.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

            for agent, value in scores.items():
                try:
                    st.write(f"**{agent} — {float(value):.1f}/10**")
                    st.progress(
                        min(100, max(0, int(float(value) * 10)))
                    )
                except (TypeError, ValueError):
                    st.write(f"**{agent}: {value}**")

        with st.expander("Read the consolidated review", expanded=True):
            st.text_area(
                "Final consolidated review",
                value=str(result.get("final_review", "")),
                height=450,
                key="full_review_text",
            )

        with st.expander("Individual specialist reviews"):
            for agent, review in result.get("agent_results", {}).items():
                with st.expander(agent):
                    st.write(str(review))

        with st.expander("Evidence verification results"):
            st.json(result.get("verification_results", {}))

        with st.expander("Plagiarism summary"):
            st.text(str(result.get("plagiarism_summary", "")))

        report_path = result.get("report_path")

        if report_path and Path(report_path).is_file():
            show_file_download(
                report_path,
                "⬇️ Download Generated Review Report",
            )


# ============================================================
# PAGE 3: PLAGIARISM CHECK
# ============================================================

elif selected_page == "🔍 Plagiarism Check":

    st.title("🔍 Plagiarism Similarity Check")

    st.write(
        "Compare your paper against local reference PDFs and optionally "
        "compare passages with arXiv abstracts."
    )

    if get_reference_directories():
        st.success("Local reference directories detected.")

        for directory in get_reference_directories():
            st.caption(directory)
    else:
        st.warning(
            "No reference-paper directory was found. Create "
            "'data/reference papers' or 'data/reference_papers' "
            "and add reference PDFs before running the check."
        )

    uploaded_path = display_uploaded_paper_selector(
        "plagiarism_upload"
    )

    use_arxiv = st.checkbox(
        "Also search arXiv abstracts",
        value=False,
        help=(
            "This sends a search query to arXiv. Only retrieved "
            "abstracts are compared; this is not a comprehensive "
            "internet-wide plagiarism check."
        ),
    )

    arxiv_query = ""

    if use_arxiv:
        arxiv_query = st.text_input(
            "Research topic or keywords for arXiv",
            placeholder="e.g. retrieval augmented generation",
        )

    if uploaded_path and st.button(
        "🔎 Run Plagiarism Check",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Comparing paper passages with the available sources..."
            ):
                result = run_plagiarism_analysis(
                    uploaded_path,
                    online_arxiv=use_arxiv,
                    arxiv_query=arxiv_query,
                )

            st.session_state.plagiarism_result = result

            save_history({
                "type": "Plagiarism Check",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "paper": uploaded_path.name,
                "status": result.get("summary", {}).get("status", ""),
                "overall_score": None,
                "report_path": result.get("txt_report", ""),
            })

            st.success("Similarity check completed.")

        except Exception as error:
            st.error(f"Plagiarism check failed: {error}")

            with st.expander("Technical details"):
                st.code(traceback.format_exc())

    result = st.session_state.plagiarism_result

    if result:
        summary = result.get("summary", {})

        st.markdown("---")
        st.subheader("Similarity Results")

        st.info(
            f"Status: {summary.get('status', 'UNKNOWN')}"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            display_metric(
                "Exact Matches",
                summary.get("exact_match_records", 0),
            )

        with c2:
            display_metric(
                "TF-IDF Matches",
                summary.get("tfidf_match_records", 0),
            )

        with c3:
            display_metric(
                "Semantic Matches",
                summary.get("semantic_match_records", 0),
            )

        with c4:
            display_metric(
                "Flagged Coverage",
                f"{summary.get('estimated_flagged_word_coverage_percent', 0)}%",
                "Approximate chunk-based estimate",
            )

        st.caption(
            f"Main-text words checked: "
            f"{summary.get('submitted_main_text_word_count', 'N/A')}"
        )

        st.warning(
            summary.get(
                "note",
                "Similarity results require human interpretation.",
            )
        )

        matches = result.get("content_matches", [])

        if matches:
            st.subheader("Potential Content Matches")

            for index, match in enumerate(matches, 1):
                with st.expander(
                    f"Match {index}: {match.get('source', 'Unknown source')} "
                    f"— {match.get('match_type', 'Unknown type')}"
                ):
                    st.write(
                        "Similarity:",
                        f"{float(match.get('similarity_score', 0)) * 100:.2f}%",
                    )

                    st.write("**Submitted passage**")
                    st.write(match.get("submitted_passage", ""))

                    st.write("**Reference passage**")
                    st.write(match.get("reference_passage", ""))

                    st.caption(
                        f"Source: {match.get('source_path', '')}"
                    )
        else:
            st.info(
                "No content matches crossed the configured thresholds "
                "in the sources checked."
            )

        bibliography_matches = result.get(
            "bibliography_matches",
            [],
        )

        if bibliography_matches:
            st.subheader("Possible Duplicate Bibliography Entries")

            st.dataframe(
                [
                    {
                        "Source": item.get("source", ""),
                        "Similarity": round(
                            float(item.get("similarity_score", 0)) * 100,
                            2,
                        ),
                        "Submitted reference": item.get(
                            "submitted_reference", ""
                        ),
                    }
                    for item in bibliography_matches
                ],
                use_container_width=True,
                hide_index=True,
            )

        arxiv_matches = result.get("arxiv_matches", [])

        if arxiv_matches:
            st.subheader("arXiv Abstract Matches")

            for index, match in enumerate(arxiv_matches, 1):
                with st.expander(
                    f"{index}. {match.get('source', 'Unknown paper')}"
                ):
                    st.write(
                        "Similarity:",
                        f"{float(match.get('similarity_score', 0)) * 100:.2f}%",
                    )
                    st.write(match.get("reference_passage", ""))
                    st.write(match.get("source_path", ""))

        reference_pdfs = result.get("reference_pdfs_checked", [])

        with st.expander("Reference PDFs checked"):
            if reference_pdfs:
                st.dataframe(
                    reference_pdfs,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No local reference PDFs were loaded.")

        show_file_download(
            result.get("txt_report", ""),
            "⬇️ Download Plagiarism Text Report",
        )

        show_file_download(
            result.get("json_report", ""),
            "⬇️ Download Plagiarism JSON Report",
        )


# ============================================================
# PAGE 4: TEMPLATE COMPLIANCE
# ============================================================

elif selected_page == "📋 Template Compliance":

    st.title("📋 Template Compliance Checker")

    st.write(
        "Evaluate a research paper against your selected template "
        "and conference-specific submission requirements."
    )

    st.info(
        "Enter the conference details and requirements provided by "
        "the conference organizers. Built-in templates are starting "
        "points and may not represent the official conference rules."
    )

    # --------------------------------------------------------
    # 1. Upload PDF
    # --------------------------------------------------------

    uploaded_path = display_uploaded_paper_selector(
        "template_upload"
    )

    # --------------------------------------------------------
    # 2. Select the paper stage (required input)
    # --------------------------------------------------------

    st.markdown("#### Paper Stage")

    submission_stage = st.selectbox(
        "Which stage does this paper belong to?",
        ["Anonymous Submission", "Camera-Ready", "Published Paper"],
        index=0,
        help=(
            "Choose Anonymous Submission for peer review, Camera-Ready for "
            "the accepted final manuscript, or Published Paper for a paper "
            "already published. This selection controls stage-specific checks."
        ),
        key="template_submission_stage",
    )

    stage_help = {
        "Anonymous Submission": "Checks for possible author-identifying details; manually verify double-blind requirements.",
        "Camera-Ready": "Prompts verification of final author details and venue-specific camera-ready requirements.",
        "Published Paper": "Prompts verification of applicable publisher, DOI, copyright, and final-version elements.",
    }
    st.caption(stage_help[submission_stage])

    # --------------------------------------------------------
    # 3. Conference details
    # --------------------------------------------------------

    st.markdown("#### Conference Details")

    conference_name = st.text_input(
        "Conference Name",
        placeholder="Enter the full conference name",
        key="template_conference_name",
    )

    guidelines_url = st.text_input(
        "Conference Guidelines URL",
        placeholder="https://official-conference-website.org/author-guidelines",
        help=(
            "Enter the official conference website page containing "
            "the paper submission or author formatting guidelines."
        ),
        key="template_guidelines_url",
    )

    # --------------------------------------------------------
    # Use an internal default template without showing the
    # Submission Template and Template Requirements controls.
    # --------------------------------------------------------

    template_name = (
        "General Research Paper"
        if "General Research Paper" in DEFAULT_TEMPLATES
        else next(iter(DEFAULT_TEMPLATES))
    )
    template = DEFAULT_TEMPLATES[template_name]
    section_list = list(template.get("required_sections", []))

    # --------------------------------------------------------
    # 6. Minimum and maximum page counts
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        min_pages = st.number_input(
            "Minimum Page Count",
            min_value=1,
            max_value=500,
            value=int(template["minimum_pages"]),
            step=1,
            key=f"minimum_pages_{template_name}",
        )

    with col2:
        max_pages = st.number_input(
            "Maximum Page Count",
            min_value=int(min_pages),
            max_value=500,
            value=max(
                int(min_pages),
                int(template["maximum_pages"]),
            ),
            step=1,
            key=f"maximum_pages_{template_name}",
        )

    # --------------------------------------------------------
    # 7. User-configurable page dimensions
    # --------------------------------------------------------

    st.markdown("#### Page Dimensions")

    dimension_col1, dimension_col2 = st.columns(2)

    with dimension_col1:
        expected_width = st.number_input(
            "Expected Page Width (inches)",
            min_value=1.0,
            max_value=30.0,
            value=float(
                template.get("expected_page_width_inches", 8.27)
            ),
            step=0.01,
            format="%.2f",
            key=f"page_width_{template_name}",
        )

    with dimension_col2:
        expected_height = st.number_input(
            "Expected Page Height (inches)",
            min_value=1.0,
            max_value=30.0,
            value=float(
                template.get("expected_page_height_inches", 11.69)
            ),
            step=0.01,
            format="%.2f",
            key=f"page_height_{template_name}",
        )

    page_size_tolerance = st.number_input(
        "Page Dimension Tolerance (inches)",
        min_value=0.0,
        max_value=2.0,
        value=float(
            template.get("page_size_tolerance_inches", 0.45)
        ),
        step=0.05,
        format="%.2f",
        help=(
            "Allowed difference between the actual PDF page dimensions "
            "and the expected dimensions."
        ),
        key=f"page_tolerance_{template_name}",
    )

    check_page_size = st.checkbox(
        "Check approximate page dimensions",
        value=True,
        key=f"check_page_size_{template_name}",
    )

    # --------------------------------------------------------
    # 8. Run compliance check
    # --------------------------------------------------------

    if st.button(
        "📑 Check Template Compliance",
        type="primary",
        use_container_width=True,
        key="run_template_compliance",
    ):

        if not uploaded_path:
            st.error("Please upload a research paper PDF.")

        elif not conference_name.strip():
            st.error("Please enter the conference name.")

        elif min_pages > max_pages:
            st.error(
                "Maximum page count must be greater than "
                "or equal to minimum page count."
            )

        else:
            try:
                with st.spinner(
                    "Checking paper structure and PDF properties..."
                ):

                    # Preserve the existing template configuration.
                    # Apply the user's page-dimension settings to this run.
                    configured_template = dict(template)

                    configured_template[
                        "expected_page_width_inches"
                    ] = float(expected_width)

                    configured_template[
                        "expected_page_height_inches"
                    ] = float(expected_height)

                    configured_template[
                        "page_size_tolerance_inches"
                    ] = float(page_size_tolerance)

                    # The checker reads DEFAULT_TEMPLATES internally.
                    # Save the original configuration and restore it
                    # immediately after the check.
                    original_template = DEFAULT_TEMPLATES[template_name]

                    try:
                        DEFAULT_TEMPLATES[template_name] = (
                            configured_template
                        )

                        result = run_template_check(
                            uploaded_path,
                            template_name,
                            section_list,
                            int(min_pages),
                            int(max_pages),
                            check_page_size,
                            submission_stage=submission_stage,
                        )

                    finally:
                        DEFAULT_TEMPLATES[template_name] = (
                            original_template
                        )

                    # Attach user-supplied conference information
                    # to this result.
                    result["conference_name"] = conference_name.strip()
                    result["guidelines_url"] = guidelines_url.strip()
                    result["template_name"] = template_name
                    result["submission_stage"] = submission_stage
                    result["configured_page_width_inches"] = float(
                        expected_width
                    )
                    result["configured_page_height_inches"] = float(
                        expected_height
                    )
                    result["configured_page_tolerance_inches"] = float(
                        page_size_tolerance
                    )

                    st.session_state.template_result = result

                    save_history({
                        "type": "Template Compliance",
                        "date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "paper": uploaded_path.name,
                        "conference_name": conference_name.strip(),
                        "guidelines_url": guidelines_url.strip(),
                        "template_name": template_name,
                        "submission_stage": submission_stage,
                        "status": result["eligibility"],
                        "overall_score": None,
                        "report_path": "",
                    })

                st.success("Template screening completed.")

            except Exception as error:
                st.error(f"Template check failed: {error}")

                with st.expander("Technical Details"):
                    st.code(traceback.format_exc())

    # --------------------------------------------------------
    # 9. Display compliance results
    # --------------------------------------------------------

    result = st.session_state.get("template_result")

    if result:

        st.markdown("---")
        st.subheader("Compliance Results")

        if result.get("conference_name"):
            st.write(
                "**Conference:**",
                result["conference_name"],
            )

        if result.get("template_name"):
            st.write(
                "**Selected Template:**",
                result["template_name"],
            )

        if result.get("submission_stage"):
            st.write(
                "**Paper Stage:**",
                result["submission_stage"],
            )

        if result.get("guidelines_url"):
            st.write(
                "**Guidelines URL:**",
                result["guidelines_url"],
            )

        eligibility = result.get(
            "eligibility",
            "NEEDS MANUAL REVIEW",
        )

        if eligibility == "REJECTED / INCOMPLETE":
            st.error(f"Status: {eligibility}")

        elif eligibility == "NEEDS MANUAL REVIEW":
            st.warning(f"Status: {eligibility}")

        else:
            st.success(f"Status: {eligibility}")

        c1, c2, c3 = st.columns(3)

        with c1:
            display_metric(
                "Pages",
                result.get("page_count", "N/A"),
            )

        with c2:
            display_metric(
                "Required Sections",
                len(section_list),
            )

        with c3:
            display_metric(
                "Conference",
                result.get("conference_name", "N/A"),
            )

        if result.get("checks"):
            st.subheader("Detailed Checks")
            st.dataframe(
                result["checks"],
                use_container_width=True,
                hide_index=True,
            )

        if result.get("missing_sections"):
            st.subheader("Missing Sections")
            for section in result["missing_sections"]:
                st.warning(section)

        if result.get("reasons"):
            st.subheader("Findings")
            for reason in result["reasons"]:
                st.write(f"• {reason}")

        st.caption(
            "Automated screening can identify potential issues, "
            "but it does not guarantee compliance with every official "
            "conference requirement or guarantee paper acceptance."
        )
       
# ============================================================
# PAGE 5: PAPER SCORE
# ============================================================

elif selected_page == "⭐ Paper Score":

    st.title("⭐ Paper Score")

    st.write(
        "View the individual agent scores and the overall score "
        "returned by the existing review pipeline."
    )

    result = st.session_state.review_result

    if not result:
        st.info(
            "No full review is available in this session. "
            "Open Complete Paper Review and run the pipeline first."
        )

        if st.button("Go to Complete Paper Review"):
            st.session_state.navigation = "📄 Complete Paper Review"
            st.rerun()

    else:
        overall_score = result.get("overall_score")
        scores = result.get("agent_scores", {})

        col1, col2 = st.columns(2)

        with col1:
            display_metric(
                "Overall Score",
                format_score(overall_score),
                "Calculated by the existing ScoreFusion component",
            )

        with col2:
            display_metric(
                "Agents Scored",
                len(scores),
                "Specialist agent results",
            )

        if scores:
            st.subheader("Agent Score Breakdown")

            for agent, score in scores.items():
                try:
                    score_float = float(score)

                    left, right = st.columns([4, 1])

                    with left:
                        st.write(f"**{agent}**")
                        st.progress(
                            min(100, max(0, int(score_float * 10)))
                        )

                    with right:
                        st.metric("Score", f"{score_float:.1f}/10")

                except (TypeError, ValueError):
                    st.write(f"{agent}: {score}")

            st.subheader("Score Table")

            st.dataframe(
                [
                    {"Agent": agent, "Score": score}
                    for agent, score in scores.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "The overall score is the value returned by the existing "
            "score-fusion implementation. It is an automated assessment, "
            "not a prediction of acceptance by a journal or conference."
        )


# ============================================================
# PAGE 6: EVIDENCE VERIFICATION
# ============================================================

elif selected_page == "🛡️ Evidence Verification":

    st.title("🛡️ Evidence Verification")

    st.write(
        "Inspect whether the specialist findings were marked SUPPORTED, "
        "NEEDS_REVISION, or UNVERIFIED by the existing verifier."
    )

    result = st.session_state.review_result

    if not result:
        st.info(
            "Run a Complete Paper Review first to generate verification results."
        )

    else:
        verification = result.get("verification_results", {})

        if not verification:
            st.warning("No verification results were returned.")
        else:
            statuses = [
                item.get("status", "UNVERIFIED")
                for item in verification.values()
            ]

            c1, c2, c3 = st.columns(3)

            with c1:
                display_metric(
                    "Supported",
                    statuses.count("SUPPORTED"),
                )

            with c2:
                display_metric(
                    "Needs Revision",
                    statuses.count("NEEDS_REVISION"),
                )

            with c3:
                display_metric(
                    "Unverified",
                    statuses.count("UNVERIFIED"),
                )

            for agent, item in verification.items():
                status = item.get("status", "UNVERIFIED")

                with st.expander(f"{agent} — {status}"):
                    st.write("**Status:**", status)

                    st.write(
                        "**Correction attempts:**",
                        item.get("attempts", 0),
                    )

                    st.write("**Feedback**")
                    st.write(
                        item.get("feedback", "No feedback supplied.")
                    )

                    unsupported = item.get("unsupported_claims", [])

                    st.write("**Claims requiring further checking**")

                    if unsupported:
                        for claim in unsupported:
                            st.markdown(f"- {claim}")
                    else:
                        st.write("No claims listed.")

                    missing = item.get("missing_evidence", [])

                    st.write("**Missing evidence**")

                    if missing:
                        for evidence in missing:
                            st.markdown(f"- {evidence}")
                    else:
                        st.write("No missing evidence listed.")

            st.warning(
                "SUPPORTED does not guarantee factual correctness. "
                "UNVERIFIED means the available evidence was insufficient "
                "for confirmation; it does not automatically mean the "
                "finding is false."
            )




# ============================================================
# PAGE 7: REPORTS AND HISTORY
# Show only results registered by Streamlit actions
# ============================================================

elif selected_page == "📊 Reports & History":

    st.title("📊 Reports & History")

    st.write(
        "View and download reports generated through the "
        "AutoReview Streamlit interface."
    )

    history = read_history()

    if not history:
        st.info(
            "No Streamlit results have been recorded yet. "
            "Run a paper review or check to see its results here."
        )

    else:
        # Display summary metrics
        full_reviews = [
            item for item in history
            if item.get("type") == "Full Paper Review"
        ]

        plagiarism_checks = [
            item for item in history
            if item.get("type") == "Plagiarism Check"
        ]

        template_checks = [
            item for item in history
            if item.get("type") == "Template Compliance"
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            display_metric(
                "Paper Reviews",
                len(full_reviews),
                "Run through Streamlit",
            )

        with col2:
            display_metric(
                "Plagiarism Checks",
                len(plagiarism_checks),
                "Run through Streamlit",
            )

        with col3:
            display_metric(
                "Template Checks",
                len(template_checks),
                "Run through Streamlit",
            )

        st.markdown("---")

        # Filter results by type
        filter_type = st.selectbox(
            "Filter results",
            [
                "All Results",
                "Full Paper Review",
                "Plagiarism Check",
                "Template Compliance",
            ],
        )

        if filter_type == "All Results":
            filtered_history = history
        else:
            filtered_history = [
                item for item in history
                if item.get("type") == filter_type
            ]

        if not filtered_history:
            st.info("No results are available for this filter.")

        else:
            st.subheader("Streamlit Run History")

            st.dataframe(
                [
                    {
                        "Date": item.get("date", ""),
                        "Check Type": item.get("type", ""),
                        "Paper": item.get("paper", ""),
                        "Score": (
                            format_score(item["overall_score"])
                            if item.get("overall_score") is not None
                            else "N/A"
                        ),
                        "Status": item.get("status", "Completed"),
                    }
                    for item in filtered_history
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")
            st.subheader("Report Downloads")

            for index, item in enumerate(filtered_history):
                paper = item.get("paper", "Unknown paper")
                check_type = item.get("type", "Unknown check")
                date = item.get("date", "Unknown date")
                report_path = item.get("report_path", "")

                with st.expander(
                    f"{check_type} | {paper} | {date}"
                ):
                    st.write("**Paper:**", paper)
                    st.write("**Check type:**", check_type)
                    st.write(
                        "**Status:**",
                        item.get("status", "Completed"),
                    )

                    if item.get("overall_score") is not None:
                        st.write(
                            "**Overall score:**",
                            format_score(item["overall_score"]),
                        )

                    if report_path and Path(report_path).is_file():
                        show_file_download(
                            report_path,
                            f"Download {Path(report_path).name}",
                        )
                    else:
                        st.info(
                            "No downloadable report path was saved "
                            "for this result."
                        )

            # Export only the Streamlit history records
            history_json = json.dumps(
                filtered_history,
                indent=2,
                ensure_ascii=False,
            )

            st.download_button(
                "⬇️ Download Filtered Streamlit History",
                data=history_json,
                file_name="autoreview_streamlit_history.json",
                mime="application/json",
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        AutoReview · AI-Powered Research Paper Review System<br>
        Automated screening supports research evaluation; final decisions
        require human judgement.
    </div>
    """,
    unsafe_allow_html=True,
)
