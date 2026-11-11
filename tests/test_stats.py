from collections import Counter

from findex.stats import collect_stats


def test_lazy_and_eager_equal(tmp_path):
    (tmp_path / "a.txt").write_text("Python python! Привіт", encoding="utf-8")
    (tmp_path / "b.txt").write_text("python DNS", encoding="utf-8")
    lazy = collect_stats(tmp_path)
    eager = collect_stats(tmp_path, eager=True)
    assert lazy.documents == eager.documents == 2
    assert lazy.tokens == eager.tokens == 5
    assert lazy.vocabulary_size == eager.vocabulary_size == 3
    assert lazy.counts == eager.counts == Counter({"python": 3, "привіт": 1, "dns": 1})


def test_limit_zero_and_one(tmp_path):
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text("One two", encoding="utf-8")
    assert collect_stats(tmp_path, limit=0).documents == 0
    assert collect_stats(tmp_path, limit=1).documents == 1
    assert collect_stats(tmp_path, limit=1).tokens == 2
