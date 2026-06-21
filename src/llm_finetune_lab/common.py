"""あちこちで使う共通の小道具。"""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import CharTokenizer
from .transformer import GPTConfig, MiniGPT

# プロジェクトのルート（このファイルから2つ上）。データやモデルの保存先の基準。
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"


def get_device() -> torch.device:
    """使える一番速い計算装置を選ぶ。Mac なら MPS、無ければ CPU。

    たとえ話: 計算という荷物運びを、台車(GPU/MPS)があれば台車で、
            無ければ手持ち(CPU)で運ぶ、の自動判定。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_corpus(name: str = "tiny_corpus.txt") -> str:
    """data/ 配下のテキストを読み込む。"""
    return (DATA_DIR / name).read_text(encoding="utf-8")


def load_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[MiniGPT, CharTokenizer, GPTConfig]:
    """保存したモデル・設定・トークナイザをまとめて復元する。

    pretrain で作った土台を、SFT / LoRA / DPO が読み込むのに使う。
    """
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    tok = CharTokenizer.from_stoi(ckpt["stoi"])
    return model, tok, config


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """学習データから (入力 x, 正解 y) のミニバッチをランダムに切り出す。

    y は x を「1文字だけ右にずらした」もの。
      x = "the ca"
      y = "he cat"   ← x の各位置の「次の文字」が正解になる。
    これで「次の文字当て」の練習問題が自動で作れる（教師データ不要＝自己教師あり）。
    """
    # 開始位置をランダムに batch_size 個選ぶ。
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
