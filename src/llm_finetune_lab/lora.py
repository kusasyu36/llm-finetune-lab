"""LoRA (Low-Rank Adaptation) をゼロから自作する。

■ 何の問題を解くの？
  巨大モデル(数十億パラメータ)を丸ごと再学習するのは、お金も時間もメモリも莫大。
  でも新しいタスクに合わせたい。そこで「元の重みは凍結(freeze)したまま、
  小さな"付け足し"だけを学習する」のが LoRA。

■ 仕組み（たとえ話）
  元の重み W は、分厚い百科事典。書き換えると大事故。
  だから百科事典はそのまま(凍結)にして、薄い「訂正メモ(ΔW)」だけ挟む。
  この訂正メモを、大きな行列1枚ではなく "細い2枚(A と B)" の掛け算で作る:

        ΔW = B @ A     （B: d×r,  A: r×d,  r はとても小さい=低ランク）

  r=4 とかにすると、学習するパラメータが元の数百分の一で済む。
  出力は:  y = x W^T  +  (x A^T) B^T * scale

■ 効果
  学習対象が激減 → 省メモリ・高速・小さな差分ファイルで配布可能。
  実務(PEFTライブラリ)でも全く同じ考え方。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """既存の nn.Linear を包んで、低ランクの訂正メモ(A,B)を足すラッパー。"""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.base = base
        self.r = r
        # scale = alpha / r。訂正メモの効き具合を調整するボリュームつまみ。
        self.scale = alpha / r

        in_f = base.in_features
        out_f = base.out_features

        # 元の重みは凍結（学習しない）。これが LoRA の肝。
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # 訂正メモを作る細い2枚。
        #   A: 入力を r 次元へ絞る（要約する）
        #   B: r 次元から出力へ広げる
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))

        # 初期化の工夫: A はランダム、B はゼロ。
        # → 学習開始時 ΔW = B@A = 0 なので「最初は元モデルと完全に同じ挙動」。
        #   そこから少しずつ訂正メモを育てるので、安全に学習が始まる。
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 元の出力（凍結された百科事典の答え）。
        base_out = self.base(x)
        # 訂正メモの出力: x → A で絞る → B で広げる → ボリューム調整。
        lora_out = (x @ self.A.t()) @ self.B.t() * self.scale
        return base_out + lora_out


def apply_lora(model: nn.Module, r: int = 4, alpha: int = 8, targets: tuple[str, ...] = ("qkv", "proj", "fc")) -> nn.Module:
    """モデル内の対象 nn.Linear を LoRALinear に差し替える。

    targets: 名前にこれらの語を含む Linear だけ改造する
             （アテンションやMLPの主要な層だけ狙い撃ち）。
    """
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in targets):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


def mark_only_lora_trainable(model: nn.Module) -> None:
    """LoRA の A,B 以外を全部凍結する（学習対象を訂正メモだけに限定）。"""
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(".A") or name.endswith(".B"))


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(学習するパラメータ数, 全パラメータ数) を返す。LoRA の省エネ具合の確認用。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
