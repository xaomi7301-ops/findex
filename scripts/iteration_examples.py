"""Short experiments to rehearse before the 5-minute defense."""

import unicodedata
from collections.abc import Iterator


class Countdown(Iterator[int]):
    """Hand-written iterator using __iter__ and __next__."""

    def __init__(self, n: int):
        self.n = n

    def __iter__(self) -> "Countdown":
        return self

    def __next__(self) -> int:
        if self.n <= 0:
            raise StopIteration
        result = self.n
        self.n -= 1
        return result


def countdown(n: int):
    while n > 0:
        yield n
        n -= 1


def main():
    xs = [1, 2, 3]
    iterator = iter(xs)
    print("Original list after next:", next(iterator), xs)
    print("Remaining items:", list(iterator))
    print("Exhausted iterator:", list(iterator))
    print("Manually implemented iterator:", list(Countdown(3)))
    generator = countdown(3)
    print("Generator first value:", next(generator))
    print("Generator remaining:", list(generator))
    left, right = "café", "cafe\u0301"
    print("Unicode strings equal before NFC?", left == right)
    print(
        "Unicode strings equal after NFC?",
        unicodedata.normalize("NFC", left) == unicodedata.normalize("NFC", right),
    )


if __name__ == "__main__":
    main()
