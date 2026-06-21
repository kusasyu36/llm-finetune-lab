"""事前学習（pretraining）。

「大量の文章を読んで、ひたすら次の文字を当てる」だけの練習。
お手本の正解ラベルを人間が用意しなくていい（文章自体が答えを持っている）ので
"自己教師あり学習" と呼ぶ。これが今のLLMの土台。

たとえ話: 子どもが大量の絵本を読むうちに、特に教わらなくても
        「むかしむかし、あるところに___」の続きが書けるようになる、あれ。

実行:
    python -m llm_finetune_lab.pretrain
保存:
    checkpoints/pretrained.pt （次の SFT / LoRA / DPO が土台として読む）
"""

from __future__ import annotations

import torch

from .common import CKPT_DIR, get_batch, get_device, load_corpus
from .tokenizer import CharTokenizer
from .transformer import GPTConfig, MiniGPT


def train_pretrain(
    steps: int = 2000,
    batch_size: int = 32,
    block_size: int = 64,
    lr: float = 3e-4,
    eval_every: int = 200,
    seed: int = 0,
) -> tuple[MiniGPT, CharTokenizer]:
    torch.manual_seed(seed)
    device = get_device()

    # 1. 文章を読み、トークナイザ(翻訳機)を作り、全文を数字列にする。
    text = load_corpus()
    tok = CharTokenizer(text)
    data = tok.encode_tensor(text)
    print(f"語彙数={tok.vocab_size} 文字数={len(data)} device={device}")

    # 2. モデルを作る。
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size)
    model = MiniGPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数={n_params:,}")

    # 3. オプティマイザ = 損失を減らす方向にパラメータを少しずつ動かす係。
    #    AdamW は今のLLM学習の定番。坂道(損失)を転がり下りるボール、のイメージ。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 4. 学習ループ: 問題を解く → 間違い(loss)を測る → 直す、を steps 回。
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad()  # 前回の勾配(直す方向)をリセット
        loss.backward()        # 誤差逆伝播: どこをどう直せばいいか計算
        optimizer.step()       # 実際にパラメータを更新

        if step % eval_every == 0 or step == 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    # 5. できあがったモデルとトークナイザを保存。
    CKPT_DIR.mkdir(exist_ok=True)
    path = CKPT_DIR / "pretrained.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "stoi": tok.stoi,
        },
        path,
    )
    print(f"保存しました: {path}")
    return model, tok


def main() -> None:
    model, tok = train_pretrain()
    # ためしに生成させてみる（"the " の続きを書かせる）。
    device = next(model.parameters()).device
    start = tok.encode_tensor("the ").unsqueeze(0).to(device)
    out = model.generate(start, max_new_tokens=60, temperature=0.8, top_k=10)
    print("---- 生成例 ----")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
