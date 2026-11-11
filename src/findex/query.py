"""Lab 03: recursive-descent Boolean query parser with positional phrases.

Grammar:
    expr   := or_expr
    or_expr := and_expr ("OR" and_expr)*
    and_expr := unary (("AND")? unary)*
    unary   := "NOT" unary | atom
    atom    := WORD | QUOTED_PHRASE | "(" expr ")"

AND binds tighter than OR; adjacent atoms mean AND; NOT is unary.
AST nodes are frozen dataclasses: structural equality tests compare trees.
"""

from __future__ import annotations

import re
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.index import posting_doc_ids
from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.object_model import Index

TOKEN_PATTERN = re.compile(r'"[^"]*"|[()]|[^\s()"]+')


class Expr:
    """Composable query expression: &, | and ~ create AST nodes."""

    def __and__(self, other: Expr) -> And:
        return And(self, other)

    def __or__(self, other: Expr) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(Expr):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return set(posting_doc_ids(index.get(self.value, [])))

    def positive_terms(self) -> tuple[str, ...]:
        return (self.value,)


@dataclass(frozen=True, slots=True)
class Phrase(Expr):
    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise ValueError("Phrase queries require --positions when building the index")
        if not self.terms:
            return set()

        by_term: list[dict[int, tuple[int, ...]]] = []
        for term in self.terms:
            posting_list = index.get(term)
            if posting_list is None:
                return set()
            if isinstance(posting_list, array):
                raise ValueError("Phrase queries cannot use packed array postings")
            by_term.append({posting.doc_id: posting.positions for posting in posting_list})
        candidates: set[int] = set(by_term[0])
        for entries in by_term[1:]:
            candidates.intersection_update(entries)

        result: set[int] = set()
        for doc_id in candidates:
            starts = set(by_term[0][doc_id])
            for offset, entries in enumerate(by_term[1:], 1):
                possible = set(entries[doc_id])
                starts = {start for start in starts if start + offset in possible}
                if not starts:
                    break
            if starts:
                result.add(doc_id)
        return result

    def positive_terms(self) -> tuple[str, ...]:
        return self.terms


@dataclass(frozen=True, slots=True)
class And(Expr):
    left: Expr
    right: Expr

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Or(Expr):
    left: Expr
    right: Expr

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return self.left.positive_terms() + self.right.positive_terms()


@dataclass(frozen=True, slots=True)
class Not(Expr):
    operand: Expr

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_lengths) - self.operand.evaluate(index)

    def positive_terms(self) -> tuple[str, ...]:
        return ()


def _lex(source: str) -> list[Expr | str]:
    tokens: list[Expr | str] = []
    end = 0
    for match in TOKEN_PATTERN.finditer(source):
        if source[end:match.start()].strip():
            raise ValueError("Malformed query: unexpected character or unmatched quote")
        word = match.group()
        if word in {"(", ")"}:
            tokens.append(word)
        elif word.startswith('"'):
            words = tuple(tokenize(word[1:-1]))
            if not words:
                raise ValueError("Empty phrase")
            tokens.append(Phrase(words))
        elif word.upper() in {"AND", "OR", "NOT"}:
            tokens.append(word.upper())
        else:
            tokens.extend(Term(term) for term in tokenize(word))
        end = match.end()
    if source[end:].strip():
        raise ValueError("Malformed query: unmatched quote")
    return tokens


class _Parser:
    def __init__(self, tokens: list[Expr | str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Expr | str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self) -> Expr | str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of query")
        self.pos += 1
        return token

    def parse_or(self) -> Expr:
        result = self.parse_and()
        while self.peek() == "OR":
            self.advance()
            result = Or(result, self.parse_and())
        return result

    def parse_and(self) -> Expr:
        result = self.parse_unary()
        while (token := self.peek()) is not None and token not in {"OR", ")"}:
            if token == "AND":
                self.advance()
                if self.peek() is None or self.peek() in {"AND", "OR", ")"}:
                    raise ValueError("Missing right operand after AND")
            result = And(result, self.parse_unary())
        return result

    def parse_unary(self) -> Expr:
        if self.peek() == "NOT":
            self.advance()
            return Not(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        token = self.advance()
        if token == "(":
            result = self.parse_or()
            if self.advance() != ")":
                raise ValueError("Missing closing parenthesis")
            return result
        if isinstance(token, Expr):
            return token
        raise ValueError(f"Unexpected token: {token!r}")


def parse(source: str) -> Expr:
    """Parse one nonempty query to an AST (AST equality is structural)."""
    tokens = _lex(source)
    if not tokens:
        raise ValueError("Empty query")
    parser = _Parser(tokens)
    result = parser.parse_or()
    if parser.peek() is not None:
        raise ValueError(f"Trailing token: {parser.peek()!r}")
    return result
