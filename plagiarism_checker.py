import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pymupdf
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIRS = [PROJECT_DIR / "data" / "reference papers", PROJECT_DIR / "data" / "reference_papers"]
OUTPUT_DIR = PROJECT_DIR / "outputs"
MIN_PASSAGE_WORDS = 25
MAX_PASSAGE_WORDS = 100
PASSAGE_OVERLAP_WORDS = 15
HIGH_SIMILARITY_THRESHOLD = 0.80
MODERATE_SIMILARITY_THRESHOLD = 0.55
SEMANTIC_SIMILARITY_THRESHOLD = 0.70
BIBLIOGRAPHY_ENTRY_THRESHOLD = 0.72
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_TIMEOUT = 25
MAX_REFERENCE_FILES = 100

REFERENCE_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:(?:\d+(?:\.\d+)*[.)]?)\s+)?(?:references|bibliography|works\s+cited|literature\s+cited|list\s+of\s+references)[ \t]*[:.]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
NEXT_SECTION_PATTERN = re.compile(
    r"^[ \t]*(?:appendix(?:\s+[A-Z0-9]+)?|supplementary\s+materials?|supplemental\s+materials?|acknowledg(?:e)?ments)[ \t]*[:.]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
REFERENCE_ENTRY_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}[a-z]?\b|\barxiv\b|\bdoi\s*:|\bvol(?:ume)?\.?\s*\d|\bpp?\.\s*\d|\bproceedings\s+of\b|\bhttps?://|\bavailable\s*:|\baccessed\s*:",
    re.IGNORECASE,
)


def extract_pdf_text(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    with pymupdf.open(str(pdf_path)) as document:
        return "\n\n".join(page.get_text("text") for page in document if page.get_text("text")).strip()


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    for old, new in {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}.items():
        text = text.replace(old, new)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text):
    return normalize_text(text).split()


def count_words(text):
    return len(tokenize_words(text))


def score_reference_candidate(remaining_text):
    lines = [line.strip() for line in remaining_text.splitlines() if line.strip()][:120]
    if not lines:
        return 0.0
    citation_lines = sum(bool(REFERENCE_ENTRY_PATTERN.search(line)) for line in lines)
    compact_lines = sum(len(line.split()) <= 45 for line in lines)
    return 0.75 * citation_lines / len(lines) + 0.25 * compact_lines / len(lines)


def split_references_section(text):
    """Split a paper into main text and references, trimming obvious non-reference tails.

    The detector deliberately trims only clear post-reference material (for example,
    prompt templates or explicitly titled appendices). Ambiguous layouts are retained
    rather than silently deleting possible references.
    """
    if not text or not text.strip():
        return "", "", False

    matches = list(REFERENCE_HEADING_PATTERN.finditer(text))
    candidates = []
    for match in matches:
        position = match.start() / max(1, len(text))
        if position < 0.20:
            continue
        remaining = text[match.end():].strip()
        if count_words(remaining) < 10:
            continue
        score = score_reference_candidate(remaining)
        candidates.append((0.75 * score + 0.25 * position, match))

    if not candidates:
        print("\n[DIAGNOSTIC] No references heading detected.")
        return text.strip(), "", False

    _, selected = max(candidates, key=lambda item: item[0])
    main_text = text[:selected.start()].strip()
    bibliography = text[selected.end():].strip()

    # First cut at explicitly titled sections that commonly follow references.
    next_section = NEXT_SECTION_PATTERN.search(bibliography)
    if next_section:
        heading = next_section.group(0).strip().lower()
        if heading.startswith(("appendix", "supplementary", "supplemental")):
            appendix = bibliography[next_section.start():].strip()
            bibliography = bibliography[:next_section.start()].strip()
            if appendix:
                main_text = (main_text + "\n\n" + appendix).strip()
        else:
            bibliography = bibliography[:next_section.start()].strip()

    # Trim clearly identifiable prompt/template material that can be embedded
    # after a bibliography in generated or compiled research PDFs.
    prompt_tail_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"\*{0,2}\s*input\s*:\s*keywords_and_descriptions\s*\*{0,2}\s*$"
        r"|[-*]\s*(?:the\s+)?[‘'\"`]?can_merge[’'\"`]?\b"
        r"|[-*]\s*[‘'\"`]?keywords_to_merge[’'\"`]?\b"
        r"|[-*]\s*[‘'\"`]?merged_keyword[’'\"`]?\b"
        r"|[-*]\s*[‘'\"`]?merged_description\b"
        r"|^\s*\{\{\s*['\"]?can_merge['\"]?\s*:"

        r"|prompt for keywords aggregation"
        r"|below are the article.s title, abstract, and introduction:"
        r"|output the keywords and summary below"
        r"|<paper_begin>"
        r"|thumbnail information:"
        r")",
    )
    prompt_tail = prompt_tail_pattern.search(bibliography)
    if prompt_tail:
        bibliography = bibliography[:prompt_tail.start()].strip()

    print("\n" + "=" * 60)
    print("REFERENCE SECTION DIAGNOSTICS")
    print("=" * 60)
    print(f"\nDetected heading: {selected.group(0).strip()!r}")
    print(f"Heading position: {selected.start()} characters")
    print(f"Total extracted words: {count_words(text)}")
    print(f"Main-text words: {count_words(main_text)}")
    print(f"Bibliography words: {count_words(bibliography)}")
    if prompt_tail:
        print("Trimmed obvious prompt/template text after bibliography.")
    print("\n--- LAST 500 CHARACTERS OF MAIN TEXT ---")
    print(main_text[-500:] if main_text else "[EMPTY]")
    print("\n--- FIRST 1,000 CHARACTERS OF BIBLIOGRAPHY ---")
    print(bibliography[:1000] if bibliography else "[EMPTY]")
    print("\n--- LAST 500 CHARACTERS OF BIBLIOGRAPHY ---")
    print(bibliography[-500:] if bibliography else "[EMPTY]")
    print("=" * 60 + "\n")

    return main_text, bibliography, True


def split_into_passages(text, min_words=MIN_PASSAGE_WORDS, max_words=MAX_PASSAGE_WORDS, overlap_words=PASSAGE_OVERLAP_WORDS):
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    passages = []
    step = max(1, max_words - overlap_words)
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            passages.append(" ".join(words))
        else:
            for start in range(0, len(words), step):
                chunk = words[start:start + max_words]
                if len(chunk) < min_words:
                    break
                passages.append(" ".join(chunk))
                if start + max_words >= len(words):
                    break
    return passages


def discover_reference_pdfs(reference_dirs=None, submitted_pdf=None):
    reference_dirs = reference_dirs or DEFAULT_REFERENCE_DIRS
    submitted = Path(submitted_pdf).resolve() if submitted_pdf else None
    found, seen = [], set()
    for directory in reference_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.pdf"):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved == submitted or resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
            if len(found) >= MAX_REFERENCE_FILES:
                return found
    return sorted(found, key=lambda p: p.name.lower())


def load_reference_papers(reference_dirs=None, submitted_pdf=None):
    references = []
    for path in discover_reference_pdfs(reference_dirs, submitted_pdf):
        try:
            full_text = extract_pdf_text(path)
            main_text, bibliography, found = split_references_section(full_text)
            item = {"name": path.name, "path": str(path), "full_text": full_text,
                    "main_text": main_text, "bibliography_text": bibliography, "references_found": found}
            references.append(item)
            print(f"Loaded reference PDF: {path.name}")
            print(f"  Total words: {count_words(full_text)} | Main text: {count_words(main_text)} | Bibliography: {count_words(bibliography)}")
            if not found:
                print("  WARNING: References heading not detected; check this PDF's split manually.")
        except Exception as exc:
            print(f"WARNING: Could not read reference PDF {path.name}: {exc}")
    return references


def find_reference_context(passage, reference_text, window=500):
    normalized_reference = normalize_text(reference_text)
    normalized_passage = normalize_text(passage)
    pos = normalized_reference.find(normalized_passage)
    if pos < 0:
        return reference_text[:window].strip()
    return normalized_reference[max(0, pos - 150):min(len(normalized_reference), pos + len(normalized_passage) + 150)].strip()


def find_exact_matches(submitted_passages, references):
    matches = []
    for idx, passage in enumerate(submitted_passages):
        normalized = normalize_text(passage)
        if len(normalized.split()) < MIN_PASSAGE_WORDS:
            continue
        for ref in references:
            source = normalize_text(ref.get("main_text", ""))
            if normalized and normalized in source:
                matches.append({"submitted_passage_index": idx, "submitted_passage": passage,
                                "source": ref["name"], "source_type": "local_pdf", "source_path": ref["path"],
                                "match_type": "EXACT_MATCH", "detection_methods": ["EXACT_MATCH"],
                                "similarity_score": 1.0, "reference_passage": find_reference_context(passage, ref.get("main_text", ""))})
    return matches


def find_similar_matches(submitted_passages, references):
    matches = []
    for ref in references:
        ref_passages = split_into_passages(ref.get("main_text", ""))
        if not ref_passages:
            continue
        try:
            vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english", max_features=100000)
            matrix = vectorizer.fit_transform(submitted_passages + ref_passages)
        except ValueError:
            continue
        submitted_matrix = matrix[:len(submitted_passages)]
        reference_matrix = matrix[len(submitted_passages):]
        sims = cosine_similarity(submitted_matrix, reference_matrix)
        for idx, row in enumerate(sims):
            if not len(row):
                continue
            best = int(row.argmax())
            score = float(row[best])
            if score < MODERATE_SIMILARITY_THRESHOLD:
                continue
            matches.append({"submitted_passage_index": idx, "submitted_passage": submitted_passages[idx],
                            "source": ref["name"], "source_type": "local_pdf", "source_path": ref["path"],
                            "match_type": "TFIDF_SIMILARITY", "detection_methods": ["TFIDF_SIMILARITY"],
                            "similarity_score": score, "reference_passage": ref_passages[best]})
    return matches


def looks_like_bibliography_entry(passage):
    lower = passage.strip().lower()
    if not lower:
        return True
    signal_count = sum(s in lower for s in ["arxiv preprint", "proceedings of", "available:", "accessed:", "doi:", "journal of", "vol.", "pp."])
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", lower))
    numbered = bool(re.match(r"^\s*(?:\[\d+\]|\d+[.)])\s+", passage))
    return (numbered and has_year) or (signal_count >= 2 and has_year)


def find_semantic_matches(submitted_passages, references):
    matches, source_passages, metadata = [], [], []
    for ref in references:
        for passage in split_into_passages(ref.get("main_text", "")):
            if not looks_like_bibliography_entry(passage):
                source_passages.append(passage)
                metadata.append(ref)
    eligible = [(i, p) for i, p in enumerate(submitted_passages) if not looks_like_bibliography_entry(p)]
    if not eligible or not source_passages:
        print("No eligible passages available for semantic comparison.")
        return matches
    print(f"Semantic comparison candidates: {len(eligible)} submitted passages, {len(source_passages)} reference passages.")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        sub_emb = model.encode([p for _, p in eligible], convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        ref_emb = model.encode(source_passages, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        sims = (sub_emb @ ref_emb.T).cpu().numpy()
    except Exception as exc:
        print(f"WARNING: Semantic matching failed: {exc}")
        return matches
    for eligible_idx, row in enumerate(sims):
        if not len(row):
            continue
        best = int(row.argmax())
        score = float(row[best])
        if score < SEMANTIC_SIMILARITY_THRESHOLD:
            continue
        original_idx, submitted = eligible[eligible_idx]
        ref = metadata[best]
        matches.append({"submitted_passage_index": original_idx, "submitted_passage": submitted,
                        "source": ref["name"], "source_type": "local_pdf", "source_path": ref["path"],
                        "match_type": "SEMANTIC_SIMILARITY", "detection_methods": ["SEMANTIC_SIMILARITY"],
                        "similarity_score": score, "reference_passage": source_passages[best]})
    return matches

# Bibliography matching: split into entries and compare entry-to-entry, not whole lists.
# Shared citation boilerplate is downweighted by removing generic terms before scoring.
GENERIC_CITATION_TERMS = {
    "arxiv", "preprint", "proceedings", "conference", "association", "computational", "linguistics",
    "journal", "volume", "vol", "pages", "page", "pp", "available", "accessed", "doi", "https", "http", "www"
}

def split_bibliography_entries(text):
    if not text or not text.strip():
        return []
    # First try paragraph-separated entries; PDF extraction often preserves these boundaries.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 2:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current = []
        for line in lines:
            starts_entry = bool(re.match(r"^\s*(?:\[\d+\]|\d+[.)])\s+", line))
            # Author/year-style references often begin with a surname followed by initials.
            starts_author_year = bool(re.match(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÿ'’.-]+(?:,|\s+et\s+al\.)", line)) and bool(re.search(r"\b(?:19|20)\d{2}\b", line))
            if current and (starts_entry or starts_author_year):
                blocks.append(" ".join(current))
                current = []
            current.append(line)
        if current:
            blocks.append(" ".join(current))
    # Join wrapped lines and discard tiny fragments.
    cleaned = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip(" \t\n•-")
        if count_words(block) >= 5 and REFERENCE_ENTRY_PATTERN.search(block):
            cleaned.append(block)
    return cleaned


def bibliography_fingerprint(entry):
    # Remove punctuation and frequent generic citation vocabulary; keep names/title/year/identifier words.
    tokens = normalize_text(entry).split()
    return " ".join(token for token in tokens if token not in GENERIC_CITATION_TERMS and not token.isdigit())


def find_bibliographic_matches(submitted_references_text, references):
    submitted_entries = split_bibliography_entries(submitted_references_text)
    if not submitted_entries:
        return []
    results = []
    for ref in references:
        source_entries = split_bibliography_entries(ref.get("bibliography_text", ""))
        if not source_entries:
            continue
        sub_fingerprints = [bibliography_fingerprint(x) for x in submitted_entries]
        src_fingerprints = [bibliography_fingerprint(x) for x in source_entries]
        valid_sub = [(i, x) for i, x in enumerate(sub_fingerprints) if len(x.split()) >= 4]
        valid_src = [(i, x) for i, x in enumerate(src_fingerprints) if len(x.split()) >= 4]
        if not valid_sub or not valid_src:
            continue
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", max_features=50000)
            matrix = vectorizer.fit_transform([x for _, x in valid_sub] + [x for _, x in valid_src])
            sims = cosine_similarity(matrix[:len(valid_sub)], matrix[len(valid_sub):])
        except ValueError:
            continue
        best_for_source = {}
        for si, row in enumerate(sims):
            if not len(row):
                continue
            ri = int(row.argmax())
            score = float(row[ri])
            # Require both a relatively high score and meaningful non-boilerplate token overlap.
            sub_i = valid_sub[si][0]
            src_i = valid_src[ri][0]
            sub_tokens = set(valid_sub[si][1].split())
            src_tokens = set(valid_src[ri][1].split())
            shared = sorted(sub_tokens & src_tokens)
            meaningful_shared = [t for t in shared if len(t) > 2]
            if score < BIBLIOGRAPHY_ENTRY_THRESHOLD or len(meaningful_shared) < 3:
                continue
            # Report the strongest submitted-entry match for each source entry to avoid noisy duplicates.
            if src_i not in best_for_source or score > best_for_source[src_i]["similarity_score"]:
                best_for_source[src_i] = {
                    "source": ref["name"], "source_type": "local_pdf", "source_path": ref["path"],
                    "match_type": "POSSIBLE_DUPLICATE_REFERENCE", "detection_methods": ["ENTRY_LEVEL_TFIDF"],
                    "similarity_score": score, "submitted_reference": submitted_entries[sub_i],
                    "source_reference": source_entries[src_i], "shared_terms": meaningful_shared[:25],
                    "submitted_reference_index": sub_i + 1, "source_reference_index": src_i + 1,
                }
        results.extend(best_for_source.values())
    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)


def search_arxiv(query, max_results=5):
    if not query or not query.strip():
        return []
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}
    try:
        response = requests.get(ARXIV_API_URL, params=params, headers={"User-Agent": "AutoReview-PlagiarismChecker/2.0"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            get_text = lambda tag: (entry.findtext(f"atom:{tag}", default="", namespaces=ns) or "").strip()
            papers.append({"title": re.sub(r"\s+", " ", get_text("title")), "abstract": re.sub(r"\s+", " ", get_text("summary")), "url": get_text("id")})
        return papers
    except Exception as exc:
        print(f"WARNING: arXiv search failed: {exc}")
        return []


def compare_against_arxiv_abstracts(submitted_passages, arxiv_papers):
    papers = [p for p in arxiv_papers if p.get("abstract", "").strip()]
    if not submitted_passages or not papers:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        sub = model.encode(submitted_passages, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        abstracts = model.encode([p["abstract"] for p in papers], convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        sims = (sub @ abstracts.T).cpu().numpy()
    except Exception as exc:
        print(f"WARNING: arXiv semantic matching failed: {exc}")
        return []
    matches = []
    for i, row in enumerate(sims):
        if not len(row):
            continue
        best = int(row.argmax())
        score = float(row[best])
        if score >= SEMANTIC_SIMILARITY_THRESHOLD:
            paper = papers[best]
            matches.append({"submitted_passage_index": i, "submitted_passage": submitted_passages[i], "source": paper["title"],
                            "source_type": "arxiv_abstract", "source_path": paper["url"], "match_type": "ARXIV_ABSTRACT_SEMANTIC_SIMILARITY",
                            "detection_methods": ["ARXIV_ABSTRACT_SEMANTIC_SIMILARITY"], "similarity_score": score, "reference_passage": paper["abstract"]})
    return matches


def combine_matches(matches):
    unique = {}
    for match in matches:
        key = (match.get("submitted_passage_index"), match.get("source"), match.get("source_type"), match.get("source_path"), match.get("match_type"))
        if key not in unique:
            unique[key] = dict(match)
        else:
            methods = unique[key].setdefault("detection_methods", [])
            for method in match.get("detection_methods", []):
                if method not in methods:
                    methods.append(method)
            unique[key]["similarity_score"] = max(unique[key].get("similarity_score", 0), match.get("similarity_score", 0))
    return sorted(unique.values(), key=lambda x: x.get("similarity_score", 0), reverse=True)


def calculate_summary(main_text, content_matches, bibliography_matches, references_found):
    total = max(1, count_words(main_text))
    passage_counts = {}
    for match in content_matches:
        idx = match.get("submitted_passage_index")
        if idx is not None:
            passage_counts[idx] = max(passage_counts.get(idx, 0), count_words(match.get("submitted_passage", "")))
    # This is an approximate chunk-based estimate; overlapping chunks can overcount.
    flagged = min(total, sum(passage_counts.values()))
    return {
        "status": "POTENTIAL SIMILARITY FOUND" if content_matches else "NO MATCHES FOUND IN CHECKED SOURCES",
        "submitted_main_text_word_count": total,
        "references_section_found": references_found,
        "content_match_records": len(content_matches),
        "exact_match_records": sum("EXACT_MATCH" in m.get("detection_methods", []) for m in content_matches),
        "tfidf_match_records": sum("TFIDF_SIMILARITY" in m.get("detection_methods", []) for m in content_matches),
        "semantic_match_records": sum("SEMANTIC_SIMILARITY" in m.get("detection_methods", []) for m in content_matches),
        "bibliographic_possible_duplicate_entries": len(bibliography_matches),
        "estimated_flagged_word_coverage_percent": round(flagged / total * 100, 2),
        "note": "Similarity is an indicator for human review, not proof of plagiarism. Only the sources checked are covered; zero matches does not prove originality.",
    }


def write_reports(submitted_pdf, summary, content_matches, bibliography_matches, arxiv_matches, references, output_dir=None):
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(submitted_pdf).stem
    txt_path = output_dir / f"{stem}_plagiarism_report_{stamp}.txt"
    json_path = output_dir / f"{stem}_plagiarism_report_{stamp}.json"
    report = {
        "submitted_pdf": str(Path(submitted_pdf).resolve()), "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary, "local_reference_matches": content_matches, "bibliographic_matches": bibliography_matches,
        "arxiv_abstract_matches": arxiv_matches,
        "reference_pdfs_checked": [{"name": r["name"], "path": r["path"], "references_section_detected": r["references_found"],
                                    "total_words": count_words(r["full_text"]), "main_text_words": count_words(r["main_text"]),
                                    "bibliography_words": count_words(r["bibliography_text"])} for r in references],
        "limitations": ["Only the listed local PDFs and any retrieved arXiv abstracts were checked.",
                        "Bibliographic matches are possible duplicate citation entries, not plagiarism findings.",
                        "Automatic reference parsing can fail on unusual PDF layouts.",
                        "Semantic similarity is not proof of copied text; human review is required.",
                        "Estimated flagged-word coverage is approximate and may overcount overlapping passages."]
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("AUTOREVIEW - PLAGIARISM SIMILARITY REPORT\n" + "=" * 72 + "\n\n")
        f.write(f"Submitted PDF: {submitted_pdf}\nGenerated: {report['generated_at']}\n\nSUMMARY\n" + "-" * 72 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\nLOCAL REFERENCE CONTENT MATCHES\n" + "-" * 72 + "\n")
        if not content_matches:
            f.write("No matches crossed the configured thresholds in checked local reference main text.\n")
        for i, m in enumerate(content_matches, 1):
            f.write(f"\nMatch {i}\nSource: {m.get('source','')}\nMatch type: {m.get('match_type','')}\nSimilarity score: {m.get('similarity_score',0)*100:.2f}%\nSource path/URL: {m.get('source_path','')}\nSubmitted passage:\n{m.get('submitted_passage','')}\nReference passage:\n{m.get('reference_passage','')}\n")
        f.write("\nPOSSIBLE DUPLICATE BIBLIOGRAPHY ENTRIES (NOT A PLAGIARISM SCORE)\n" + "-" * 72 + "\n")
        if not bibliography_matches:
            f.write("No individual reference-entry pairs crossed the configured threshold.\n")
        for i, m in enumerate(bibliography_matches, 1):
            f.write(f"\nEntry pair {i}\nSource PDF: {m['source']}\nEntry similarity: {m['similarity_score']*100:.2f}%\nSubmitted reference #{m['submitted_reference_index']}:\n{m['submitted_reference']}\nSource reference #{m['source_reference_index']}:\n{m['source_reference']}\nShared meaningful terms: {', '.join(m['shared_terms'])}\n")
        f.write("\nARXIV ABSTRACT MATCHES\n" + "-" * 72 + "\n")
        if not arxiv_matches:
            f.write("No abstract-level matches reported; this does not establish that no related paper exists.\n")
        for i, m in enumerate(arxiv_matches, 1):
            f.write(f"\nMatch {i}\nSource: {m['source']}\nScore: {m['similarity_score']*100:.2f}%\nURL: {m['source_path']}\nSubmitted passage:\n{m['submitted_passage']}\nAbstract:\n{m['reference_passage']}\n")
        f.write("\nREFERENCE PDFs CHECKED\n" + "-" * 72 + "\n")
        for r in references:
            f.write(f"{r['name']} | main text: {count_words(r['main_text'])} words | bibliography: {count_words(r['bibliography_text'])} words | references heading detected: {r['references_found']}\n")
        f.write("\nDISCLAIMER\n" + "-" * 72 + "\nSimilarity results require human interpretation. Bibliographic overlaps are not counted as copied main-body content. This tool does not replace a professional plagiarism-detection service.\n")
    return txt_path, json_path


def run_plagiarism_check(submitted_pdf, reference_dirs=None, offline=False, arxiv_query=None, arxiv_results=5):
    submitted_pdf = Path(submitted_pdf)
    print("\n" + "=" * 64 + "\nAUTOREVIEW PLAGIARISM CHECKER\n" + "=" * 64)
    print(f"\nSubmitted PDF: {submitted_pdf}")
    print("\n[1/7] Extracting submitted paper text...")
    full_text = extract_pdf_text(submitted_pdf)
    if not full_text:
        raise ValueError("No text could be extracted. The PDF may be scanned/image-only and may require OCR.")
    main_text, bibliography, refs_found = split_references_section(full_text)
    print(f"Total extracted words: {count_words(full_text)}")
    print(f"Main-text words: {count_words(main_text)}")
    print(f"Bibliography words: {count_words(bibliography)}")
    print("References section detected." if refs_found else "WARNING: References heading not found; full text treated as main text.")
    print("\n[2/7] Splitting main text into passages...")
    submitted_passages = split_into_passages(main_text)
    print(f"Main-text passages created: {len(submitted_passages)}")
    if not submitted_passages:
        raise ValueError("No suitable main-text passages were created. Check the PDF text extraction and section split.")
    print("\n[3/7] Loading local reference PDFs...")
    references = load_reference_papers(reference_dirs, submitted_pdf)
    print(f"Reference PDFs loaded: {len(references)}")
    print("\n[4/7] Comparing main content against local PDFs...")
    exact = find_exact_matches(submitted_passages, references)
    print(f"Exact matches: {len(exact)}")
    tfidf = find_similar_matches(submitted_passages, references)
    print(f"TF-IDF matches: {len(tfidf)}")
    semantic = find_semantic_matches(submitted_passages, references)
    print(f"Semantic matches: {len(semantic)}")
    content = combine_matches(exact + tfidf + semantic)
    print(f"Combined content match records: {len(content)}")
    print("\n[5/7] Checking bibliography entries individually...")
    bibliography_matches = find_bibliographic_matches(bibliography, references)
    print(f"Possible duplicate bibliography entry pairs: {len(bibliography_matches)}")
    print("\n[6/7] Checking arXiv abstracts...")
    arxiv_matches = []
    if offline:
        print("Offline mode enabled; skipping arXiv search.")
    elif arxiv_query:
        print(f"arXiv query: {arxiv_query}")
        papers = search_arxiv(arxiv_query, max_results=max(1, arxiv_results))
        print(f"arXiv records retrieved: {len(papers)}")
        arxiv_matches = compare_against_arxiv_abstracts(submitted_passages, papers) if papers else []
        print(f"arXiv abstract matches: {len(arxiv_matches)}")
        time.sleep(1)
    else:
        print("No arXiv query provided; skipping online abstract search.")
    print("\n[7/7] Calculating summary and writing reports...")
    summary = calculate_summary(main_text, content, bibliography_matches, refs_found)
    txt_path, json_path = write_reports(submitted_pdf, summary, content, bibliography_matches, arxiv_matches, references)
    print("\n" + "=" * 64 + "\nPLAGIARISM CHECK COMPLETE\n" + "=" * 64)
    print(f"Status: {summary['status']}")
    print(f"Estimated flagged-word coverage: {summary['estimated_flagged_word_coverage_percent']}%")
    print(f"Exact match records: {summary['exact_match_records']}")
    print(f"TF-IDF match records: {summary['tfidf_match_records']}")
    print(f"Semantic match records: {summary['semantic_match_records']}")
    print(f"Possible duplicate bibliography entry pairs: {summary['bibliographic_possible_duplicate_entries']}")
    print(f"\nTXT report: {txt_path.resolve()}\nJSON report: {json_path.resolve()}")
    return {
        "summary": summary,
        "content_matches": content,
        "bibliography_matches": bibliography_matches,
        "arxiv_matches": arxiv_matches,
        "reference_pdfs_checked": [
            {"name": reference.get("name", "Unknown reference"),
             "path": reference.get("path", "")}
            for reference in references
        ],
        "txt_report": str(txt_path.resolve()),
        "json_report": str(json_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description="AutoReview plagiarism similarity checker. Compares a submitted PDF against local reference PDFs and optionally arXiv abstracts.")
    parser.add_argument("pdf", help="Path to the submitted research paper PDF.")
    parser.add_argument("--references", nargs="*", default=None, help="Directories containing reference PDFs.")
    parser.add_argument("--offline", action="store_true", help="Skip online arXiv abstract searching.")
    parser.add_argument("--arxiv-query", default=None, help="Topic or keywords to search on arXiv; only abstracts are compared.")
    parser.add_argument("--arxiv-results", type=int, default=5, help="Maximum number of arXiv records to retrieve.")
    args = parser.parse_args()
    dirs = [Path(p) for p in args.references] if args.references else DEFAULT_REFERENCE_DIRS
    try:
        run_plagiarism_check(args.pdf, reference_dirs=dirs, offline=args.offline, arxiv_query=args.arxiv_query, arxiv_results=max(1, args.arxiv_results))
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
