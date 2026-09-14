"""
Preprocessing utilities for Hindi legal text.

Handles:
- Unicode normalization (Devanagari has multiple encodings for the same glyph)
- Removing boilerplate court-document noise (page numbers, repeated headers)
- Optional stopword removal
- Keeping legal-relevant tokens (section numbers, "धारा 302", "FIR", etc.) intact
"""

import re
import unicodedata

# Small starter Hindi stopword list. Extend with a fuller list (e.g. from
# indic-nlp-library or stopwords-iso) for production use.
HINDI_STOPWORDS = {
    "और", "का", "की", "के", "है", "हैं", "था", "थे", "थी", "को", "में",
    "से", "पर", "यह", "वह", "एक", "इस", "उस", "कि", "जो", "भी", "ने",
    "तो", "ही", "या", "गया", "गई", "गए", "किया", "करने", "हो", "रहा",
    "रही", "रहे", "साथ", "लिए", "द्वारा", "तथा", "एवं",
}

# Patterns commonly found as noise in scanned/OCR'd Indian court documents.
NOISE_PATTERNS = [
    r"पृष्ठ\s*संख्या\s*[:\-]?\s*\d+",   # "page number: N"
    r"Page\s*No\.?\s*\d+",
    r"-{3,}",                            # long dashes from OCR
    r"_{3,}",
    r"\s{2,}",                           # collapse multi-spaces (last)
]


def normalize_unicode(text: str) -> str:
    """Normalize Devanagari unicode to a consistent composed form (NFC)."""
    return unicodedata.normalize("NFC", text)


def remove_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS[:-1]:
        text = re.sub(pattern, " ", text)
    text = re.sub(NOISE_PATTERNS[-1], " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: set = HINDI_STOPWORDS) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in stopwords)


def clean_text(text: str, drop_stopwords: bool = False) -> str:
    """Full cleaning pipeline applied before vectorization/tokenization."""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = remove_noise(text)
    if drop_stopwords:
        text = remove_stopwords(text)
    return text


if __name__ == "__main__":
    sample = "पृष्ठ संख्या: 3   आवेदक द्वारा प्रस्तुत जमानत आवेदन खारिज किया गया।"
    print("Before:", sample)
    print("After :", clean_text(sample, drop_stopwords=True))
