"""llm-finetune-lab: 大規模言語モデルの仕組みを「ゼロから自前実装」で理解する道場。

このパッケージは、巨大なライブラリやモデルのダウンロードに頼らず、
素の PyTorch だけで以下を小さく自作する:

- tokenizer.py   : 文字を数字に変える「翻訳機」
- transformer.py : 文章を読むAIの心臓部「Transformer」
- pretrain.py    : 大量の文章で「次の文字当て」を練習させる事前学習
- lora.py        : 巨大モデルを少しだけ賢く直す省エネ改造「LoRA」
- sft.py         : お手本を真似させる教師ありファインチューニング(SFT)
- dpo.py         : 「こっちの方が好き」を教える好み学習(DPO)
- prompting.py   : 学習済みモデルへの「聞き方」の工夫(プロンプト技法)
- evaluate.py    : できあがったモデルの「成績の測り方"
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
