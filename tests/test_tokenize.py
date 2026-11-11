"""Ten tokenization tests, including required Unicode edge cases."""

from findex.tokenize import tokenize


def test_mixed_case():
    assert list(tokenize("PyThOn PYTHON")) == ["python", "python"]


def test_cyrillic_and_ukrainian():
    assert list(tokenize("ПрИвІт УКРАЇНО!")) == ["привіт", "україно"]


def test_combining_accent_normalization():
    assert list(tokenize("cafe\u0301 café")) == ["café", "café"]


def test_punctuation():
    assert list(tokenize("Hello, world! (Python).")) == ["hello", "world", "python"]


def test_empty_string():
    assert list(tokenize("")) == []


def test_apostrophe_variants_are_unified():
    assert list(tokenize("п'ять п’ять пʼять don't")) == [
        "п'ять", "п'ять", "п'ять", "don't"
    ]


def test_hyphens_and_dashes_split_words():
    assert list(tokenize("client-server івано-франківськ test—case")) == [
        "client", "server", "івано", "франківськ", "test", "case"
    ]


def test_numbers_and_alphanumeric_identifiers():
    assert list(tokenize("192.168.1.1 IPv6 python3 2026")) == [
        "192", "168", "1", "1", "ipv6", "python3", "2026"
    ]


def test_casefold_better_than_lower():
    assert list(tokenize("Straße STRASSE")) == ["strasse", "strasse"]


def test_underscores_split():
    assert list(tokenize("hello_world test_case")) == ["hello", "world", "test", "case"]
