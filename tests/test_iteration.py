"""Demonstrate that both required APIs use yield-based generators."""

import inspect

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def test_required_functions_are_generators():
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgeneratorfunction(tokenize)
    assert iter(tokenize("abc")) is not None
