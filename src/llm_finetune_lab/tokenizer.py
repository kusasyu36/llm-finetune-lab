"""文字レベル トークナイザ（tokenizer = 翻訳機）。

AI（ニューラルネット）は文字をそのまま読めない。読めるのは「数字」だけ。
そこで「文字 ⇄ 数字」を変換する辞書を作る。これがトークナイザ。

たとえ話: 外国人に日本語を見せても分からないので、
        「あ=1, い=2, う=3 ...」という早見表を作って数字で渡す感じ。

ここでは一番シンプルな「1文字＝1トークン」方式にする。
（本物のGPTは「サブワード」というもう少し賢い切り方を使うが、考え方は同じ。）
"""

from __future__ import annotations

import torch


class CharTokenizer:
    """文字 ⇄ 整数ID を相互変換する最小のトークナイザ。"""

    def __init__(self, text: str) -> None:
        # 与えられた文章に「出てくる文字の種類」を全部集めて、並べる。
        # 例: "abcab" → 使われている文字は {a, b, c} の3種類。
        # sorted で並び順を固定 → 毎回同じIDになり再現性が出る。
        chars = sorted(set(text))

        # stoi = string to integer（文字→番号の早見表）。
        # itos = integer to string（番号→文字の早見表）。逆引き用。
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> "CharTokenizer":
        """保存済みの早見表(stoi)から復元する。学習時と同じIDを使うために必要。"""
        tok = cls.__new__(cls)
        tok.stoi = dict(stoi)
        tok.itos = {i: ch for ch, i in stoi.items()}
        return tok

    @property
    def vocab_size(self) -> int:
        """語彙数 = 知っている文字の種類数。モデルの出力サイズに直結する。"""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """文章 → 番号の列（エンコード = 符号化）。"""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """番号の列 → 文章（デコード = 復号。encode の逆）。"""
        return "".join(self.itos[int(i)] for i in ids)

    def encode_tensor(self, text: str) -> torch.Tensor:
        """エンコード結果を、PyTorch が食べやすい long 型テンソルにして返す。"""
        return torch.tensor(self.encode(text), dtype=torch.long)
