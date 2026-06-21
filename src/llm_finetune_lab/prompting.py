"""プロンプティング（prompting）。学習済みモデルへの"聞き方"の工夫。

モデルの中身を学習し直さなくても、入力(プロンプト)の書き方を変えるだけで
出力は大きく変わる。代表的な技を、自作モデルで体感する。

■ ここで見せる技
  1. zero-shot   : 例を見せず、いきなりお願いする
  2. few-shot    : お手本を数個見せてから、本番を解かせる（その場で学ぶ "文脈内学習"）
  3. temperature : 出力の"大胆さ"つまみ（低い=堅実 / 高い=奔放）
  4. top_k       : 候補を上位k個に絞り、変な暴走を防ぐ

実行:
    python -m llm_finetune_lab.prompting   （先に pretrain を実行しておくこと）
"""

from __future__ import annotations

import torch

from .common import CKPT_DIR, get_device, load_checkpoint


def complete(model, tok, prompt: str, device, max_new_tokens: int = 40,
             temperature: float = 0.8, top_k: int | None = 10) -> str:
    """prompt の続きをモデルに書かせて、全体を文字列で返す。"""
    ids = tok.encode_tensor(prompt).unsqueeze(0).to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main() -> None:
    device = get_device()
    path = CKPT_DIR / "pretrained.pt"
    if not path.exists():
        print("先に `python -m llm_finetune_lab.pretrain` を実行してください。")
        return

    model, tok, _ = load_checkpoint(path, device)
    torch.manual_seed(0)

    print("■ 1. zero-shot（いきなり聞く）")
    print("  ", repr(complete(model, tok, "the cat ", device)))

    print("\n■ 2. few-shot（お手本を見せてから聞く）")
    few = "the cat sat on the mat.\nthe dog sat on the log.\nthe cat "
    print("  ", repr(complete(model, tok, few, device)))

    print("\n■ 3. temperature の効果（同じ入力 'the '）")
    for t in (0.2, 0.8, 1.5):
        torch.manual_seed(0)
        print(f"  temp={t}: ", repr(complete(model, tok, "the ", device, temperature=t)))

    print("\n■ 4. top_k の効果（同じ入力 'the '）")
    for k in (1, 5, None):
        torch.manual_seed(0)
        print(f"  top_k={k}: ", repr(complete(model, tok, "the ", device, top_k=k)))


if __name__ == "__main__":
    main()
