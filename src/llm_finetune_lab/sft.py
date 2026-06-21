"""SFT (Supervised Fine-Tuning / 教師ありファインチューニング)。

事前学習したモデルは「文章を続ける」のは得意だが、まだ"指示に従う"形を
知らない。そこで「指示 → 望ましい返答」のお手本ペアを見せて真似させる。
これが SFT。ChatGPT を"おしゃべり"にした最初の工程にあたる。

■ 大事な工夫: プロンプト部分は採点しない（loss マスク）
  学習で測るのは「返答(response)をどれだけ上手に書けたか」だけ。
  指示(prompt)部分の予測ミスは数えない（そこは"問題文"であって"答え"ではない）。
  テストで、問題文の書き写しは採点せず、解答欄だけ採点するのと同じ。

■ ここでは LoRA を使って SFT する（省メモリ）。
実行:
    python -m llm_finetune_lab.sft   （先に pretrain を実行しておくこと）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import CKPT_DIR, get_device, load_checkpoint
from .lora import apply_lora, count_trainable, mark_only_lora_trainable
from .tokenizer import CharTokenizer

# 指示→返答 のお手本（コーパスに出てくる文字だけで作る）。
SFT_PAIRS: list[tuple[str, str]] = [
    ("the cat ", "sat on the mat."),
    ("the dog ", "ran to the cat."),
    ("the sun ", "is hot."),
    ("the moon ", "is cool."),
    ("i like ", "the cat and the dog."),
    ("run cat ", "run to the mat."),
]


def make_example(
    prompt: str,
    response: str,
    tok: CharTokenizer,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1ペアを (入力x, 正解y) に変換。prompt 部分の正解は -100 で「採点対象外」にする。

    PyTorch の cross_entropy は ignore_index=-100 の位置を損失計算から除外する。
    """
    full = prompt + response
    ids = tok.encode(full)[: block_size + 1]

    x = torch.tensor(ids[:-1], dtype=torch.long)
    y = torch.tensor(ids[1:], dtype=torch.long)

    # prompt の長さ分だけ y を -100 にして採点対象外に。
    prompt_len = len(tok.encode(prompt))
    y[: max(prompt_len - 1, 0)] = -100

    return x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)


def train_sft(steps: int = 800, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    device = get_device()

    # 1. 事前学習済みモデルを土台として読み込む。
    model, tok, config = load_checkpoint(CKPT_DIR / "pretrained.pt", device)

    # 2. LoRA を挿し、訂正メモ(A,B)だけを学習対象にする。
    apply_lora(model, r=4, alpha=8)
    mark_only_lora_trainable(model)
    model.to(device)
    trainable, total = count_trainable(model)
    print(f"学習パラメータ {trainable:,} / 全 {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # 3. お手本ペアを順番に学習。
    model.train()
    for step in range(1, steps + 1):
        prompt, response = SFT_PAIRS[step % len(SFT_PAIRS)]
        x, y = make_example(prompt, response, tok, config.block_size, device)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # 4. 保存。
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / "sft.pt"
    torch.save(
        {"model_state": model.state_dict(), "config": config.__dict__, "stoi": tok.stoi},
        path,
    )
    print(f"保存しました: {path}")
    return model, tok


def main() -> None:
    model, tok = train_sft()
    device = next(model.parameters()).device
    print("---- SFT後の応答例 ----")
    for prompt, _ in SFT_PAIRS[:3]:
        start = tok.encode_tensor(prompt).unsqueeze(0).to(device)
        out = model.generate(start, max_new_tokens=20, temperature=0.5, top_k=5)
        print(f"{prompt!r} -> {tok.decode(out[0].tolist())!r}")


if __name__ == "__main__":
    main()
