import pymupdf


def extract_text(pdf_path):
    """
    Extract text from a research paper PDF.
    """

    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text("text")

    document.close()

    return text


if __name__ == "__main__":

    pdf_path = "data/sample papers/paper.pdf"

    extracted_text = extract_text(pdf_path)

    print("=" * 60)
    print("Research Paper Successfully Loaded")
    print("=" * 60)

    print(f"\nTotal Characters : {len(extracted_text)}")
    print(f"Total Words : {len(extracted_text.split())}")

    print("\nFirst 1000 Characters\n")
    print(extracted_text[:1000])