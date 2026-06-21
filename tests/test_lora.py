"""LoRA のテスト。差し替え・凍結・初期一致・省パラメータを確認。"""

from __future__ import annotations

import torch
import torch.nn as nn

from llm_finetune_lab.lora import (
    LoRALinear,
    apply_lora,
    count_trainable,
    mark_only_lora_trainable,
)
from llm_finetune_lab.transformer import GPTConfig, MiniGPT


def test_lora_starts_as_identity():
    # 学習前は B=0 なので、LoRA を挟んでも出力は元の Linear と完全一致する。
    torch.manual_seed(0)
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(base(x), wrapped(x), atol=1e-6)


def test_base_weight_is_frozen():
    base = nn.Linear(8, 8)
    wrapped = LoRALinear(base, r=2)
    assert wrapped.base.weight.requires_grad is False
    assert wrapped.A.requires_grad is True
    assert wrapped.B.requires_grad is True


def test_apply_lora_reduces_trainable_params():
    config = GPTConfig(vocab_size=10, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = MiniGPT(config)
    _, total_before = count_trainable(model)

    apply_lora(model, r=4, alpha=8)
    mark_only_lora_trainable(model)
    trainable_after, total_after = count_trainable(model)

    # 学習対象がぐっと減っている（省メモリの本質）。
    assert trainable_after < total_before * 0.5
    assert trainable_after > 0
