
from core.section_identifier import identify_sections


sample_text = """
Abstract
This paper introduces a research framework.

1 Introduction
The problem is described here.

Related Work
Several earlier methods are discussed.

2. Methods
The proposed method is explained.

Results
The first results are presented.

Discussion
The findings are discussed.

Conclusion
The work is summarized.

References
Reference one.

abstract
A repeated abstract heading for testing.
"""


sections = identify_sections(sample_text)

print("\nSECTION IDENTIFIER TEST")
print("=" * 50)

for heading, content in sections.items():
    print(f"\nSection: {heading}")
    print(f"Content preview: {content[:100]}")

print("\nDetected section names:")
print(list(sections.keys()))

assert "Abstract" in sections
assert "Introduction" in sections
assert "Methodology" in sections
assert "Results" in sections
assert "References" in sections
assert "abstract" not in sections

print("\nSection normalization test passed.")
