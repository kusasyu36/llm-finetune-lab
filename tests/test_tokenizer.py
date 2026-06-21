"""トークナイザのテスト。"""

from __future__ import annotations

from llm_finetune_lab.tokenizer import CharTokenizer


def test_roundtrip():
    # encode → decode で元に戻る（往復しても壊れない）。
    tok = CharTokenizer("the cat sat")
    text = "cat"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_is_unique_chars():
    tok = CharTokenizer("aabbc")
    assert tok.vocab_size == 3  # a, b, c の3種類


def test_from_stoi_restores_mapping():
    tok = CharTokenizer("the cat")
    restored = CharTokenizer.from_stoi(tok.stoi)
    assert restored.encode("cat") == tok.encode("cat")
