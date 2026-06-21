"""DPO (Direct Preference Optimization / 直接選好最適化)。

SFT は「正解の1つ」を真似させる。でも人間の好みは「AよりBが好き」という
"比較"で表れることが多い。DPO は (指示, 良い返答, 悪い返答) の3点組を見せて、
「良い方の確率を上げ、悪い方の確率を下げる」ようにモデルを動かす。

■ ポイント: 暴走しないように"基準モデル(reference)"と比べる
  学習中のモデル(policy)が、元のSFTモデル(ref)から離れすぎないよう手綱を引く。
  βで手綱の強さを調整。これで「好みは反映するが、別人になってしまわない」。

■ 損失（やさしい言い換え）
  良い返答が ref より相対的に上手くなり、悪い返答が ref より下手になるほど
  損失が下がる。式:
    loss = -log σ( β * ( (chosenの伸び) - (rejectedの伸び) ) )
  "伸び" = log確率(policy) - log確率(ref)。

実行:
    python -m llm_finetune_lab.dpo   （先に pretrain を実行しておくこと）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import CKPT_DIR, get_device, load_checkpoint
from .lora import apply_lora, count_trainable, mark_only_lora_trainable
from .tokenizer import CharTokenizer
from .transformer import MiniGPT

# (指示, 好ましい返答, 好ましくない返答)
DPO_TRIPLES: list[tuple[str, str, str]] = [
    ("the cat ", "sat on the mat.", "the the the the the"),
    ("the dog ", "ran to the cat.", "dog dog dog dog dog"),
    ("the sun ", "is hot.", "is is is is is is."),
    ("i like ", "the cat and the dog.", "i i i i i i i i i"),
]


def response_logprob(
    model: MiniGPT,
    prompt: str,
    response: str,
    tok: CharTokenizer,
    device: torch.device,
) -> torch.Tensor:
    """返答(response)部分のトークンの「log確率の合計」を返す。

    値が大きい(0に近い)ほど、モデルはその返答を"書きやすい"と思っている。
    """
    ids = tok.encode(prompt + response)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)

    logits, _ = model(x)                       # (1, T, vocab)
    logp = F.log_softmax(logits[0], dim=-1)     # 各位置の全文字の log確率
    token_logp = logp[range(len(targets)), targets]  # 正解文字の log確率だけ抜く

    prompt_len = len(tok.encode(prompt))
    # 返答部分だけ合計（prompt 部分は手本でも何でもないので除外）。
    return token_logp[max(prompt_len - 1, 0):].sum()


def train_dpo(steps: int = 600, lr: float = 5e-4, beta: float = 0.1, seed: int = 0):
    torch.manual_seed(seed)
    device = get_device()

    # policy = これから好みを学ぶモデル（LoRA で軽く学習）。
    policy, tok, config = load_checkpoint(CKPT_DIR / "pretrained.pt", device)
    apply_lora(policy, r=4, alpha=8)
    mark_only_lora_trainable(policy)
    policy.to(device)

    # reference = 凍結した基準モデル（手綱の支点。学習しない）。
    ref, _, _ = load_checkpoint(CKPT_DIR / "pretrained.pt", device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    trainable, total = count_trainable(policy)
    print(f"学習パラメータ {trainable:,} / 全 {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        (p for p in policy.parameters() if p.requires_grad), lr=lr
    )

    policy.train()
    for step in range(1, steps + 1):
        prompt, chosen, rejected = DPO_TRIPLES[step % len(DPO_TRIPLES)]

        # policy / ref それぞれで、良い返答・悪い返答の log確率を測る。
        pi_chosen = response_logprob(policy, prompt, chosen, tok, device)
        pi_rejected = response_logprob(policy, prompt, rejected, tok, device)
        with torch.no_grad():
            ref_chosen = response_logprob(ref, prompt, chosen, tok, device)
            ref_rejected = response_logprob(ref, prompt, rejected, tok, device)

        # "伸び" = policy が ref よりどれだけ上手くなったか。
        chosen_gain = pi_chosen - ref_chosen
        rejected_gain = pi_rejected - ref_rejected

        # 良い方の伸び > 悪い方の伸び、を大きくしたい。
        loss = -F.logsigmoid(beta * (chosen_gain - rejected_gain))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} "
                  f"| chosen_gain {chosen_gain.item():+.3f} rejected_gain {rejected_gain.item():+.3f}")

    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / "dpo.pt"
    torch.save(
        {"model_state": policy.state_dict(), "config": config.__dict__, "stoi": tok.stoi},
        path,
    )
    print(f"保存しました: {path}")
    return policy, tok


def main() -> None:
    train_dpo()


if __name__ == "__main__":
    main()
