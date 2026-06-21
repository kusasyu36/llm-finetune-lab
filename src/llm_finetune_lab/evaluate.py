"""評価（evaluation）。モデルの"賢さ"を数字で測る。

作っただけでは良し悪しが分からない。客観的なものさしが要る。
ここでは言語モデルの定番 "perplexity(パープレキシティ=困惑度)" を使う。

■ perplexity とは（たとえ話）
  次の文字を当てるとき、モデルが「だいたい何択で迷っているか」の平均。
    perplexity = 1 → 1択。完璧に当てている（迷い無し）。
    perplexity = 26 → アルファベット全部からあてずっぽう（最悪）。
  小さいほど賢い。式は loss(交差エントロピー)の指数: ppl = exp(loss)。

実行:
    python -m llm_finetune_lab.evaluate
"""

from __future__ import annotations

import math

import torch

from .common import CKPT_DIR, get_batch, get_device, load_checkpoint, load_corpus


@torch.no_grad()
def perplexity(model, data: torch.Tensor, block_size: int, device: torch.device,
               batches: int = 50, batch_size: int = 32) -> float:
    """データに対する平均 perplexity を測る。"""
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


@torch.no_grad()
def next_char_accuracy(model, data: torch.Tensor, block_size: int, device: torch.device,
                       batches: int = 50, batch_size: int = 32) -> float:
    """次の文字を「1位の予想」で当てられた割合（top-1 正解率）。"""
    model.eval()
    correct = 0
    count = 0
    for _ in range(batches):
        x, y = get_batch(data, block_size, batch_size, device)
        logits, _ = model(x)
        pred = logits.argmax(dim=-1)   # 各位置で一番スコアの高い文字
        correct += (pred == y).sum().item()
        count += y.numel()
    return correct / count


def main() -> None:
    device = get_device()
    text = load_corpus()

    path = CKPT_DIR / "pretrained.pt"
    if not path.exists():
        print("先に `python -m llm_finetune_lab.pretrain` を実行してください。")
        return

    model, tok, config = load_checkpoint(path, device)
    data = tok.encode_tensor(text).to(device)

    ppl = perplexity(model, data, config.block_size, device)
    acc = next_char_accuracy(model, data, config.block_size, device)
    print(f"perplexity = {ppl:.3f}  (1に近いほど良い / 語彙数={tok.vocab_size})")
    print(f"next-char accuracy = {acc*100:.1f}%")


if __name__ == "__main__":
    main()
