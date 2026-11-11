"""Lazy Unicode tokenizer for English and Ukrainian technical notes.

Rules:
* Normalize text to NFC, then casefold (and NFC again).
* Keep inner straight/curly/modifier apostrophes, normalizing them to ASCII '.
* Hyphen and dash split words into separate tokens.
* Keep digits and alphanumeric tokens (e.g., 192, ipv6, python3).
* Underscores and all other punctuation split tokens.
* Unicode word letters are supported, not just English ASCII.
"""

import re
import unicodedata
from collections.abc import Iterator

# [^\W_] means a Unicode word character other than underscore.
# Apostrophes are allowed only between two non-empty word segments.
TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", flags=re.UNICODE)
APOSTROPHE_VARIANTS = str.maketrans({"\u2019": "'", "\u02bc": "'", "\u2018": "'"})


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens one by one using re.finditer.

    NFC creates a temporary string for each document, not a list of tokens.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = unicodedata.normalize("NFC", normalized).translate(APOSTROPHE_VARIANTS)
    for match in TOKEN_RE.finditer(normalized):
        yield match.group()
