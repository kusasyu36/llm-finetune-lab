"""Transformer のテスト。形が合うか・学習で損失が下がるか・因果性を確認。"""

from __future__ import annotations

import torch

from llm_finetune_lab.transformer import GPTConfig, MiniGPT


def _tiny_model():
    config = GPTConfig(vocab_size=10, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    return MiniGPT(config), config


def test_forward_shapes():
    model, config = _tiny_model()
    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    logits, loss = model(x, x)
    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss.dim() == 0  # スカラー


def test_generate_extends_sequence():
    model, config = _tiny_model()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=5)
    assert out.shape == (1, 6)  # 1 + 5


def test_loss_decreases_on_repeating_pattern():
    # 同じパターンを繰り返し学習させると損失が下がる（学習できている証拠）。
    torch.manual_seed(0)
    model, config = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    x, y = data[:, :-1], data[:, 1:]

    first = None
    for step in range(50):
        _, loss = model(x, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_causal_no_future_leak():
    # 因果マスクの確認: ある位置の出力は「未来の入力」を変えても変わらない。
    torch.manual_seed(0)
    model, config = _tiny_model()
    model.eval()
    x = torch.randint(0, config.vocab_size, (1, config.block_size))

    with torch.no_grad():
        logits_a, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % config.vocab_size  # 最後の文字だけ変える
        logits_b, _ = model(x2)

    # 最後の位置を変えても、その前の位置の予測は不変であるべき。
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
