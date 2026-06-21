# llm-finetune-lab

大規模言語モデル（LLM）の中身を **「ゼロから自前実装」** で理解するための学習用リポジトリ。
巨大ライブラリや事前学習済みモデルのダウンロードに頼らず、**素の PyTorch** だけで
LLM の一生（事前学習 → ファインチューニング → 好み学習 → 評価 → 使い方）を一通り再現する。

- GPU 不要。Mac（MPS）でも CPU でも数分で全工程が回る。
- すべて1ファイル＝1テーマ。日本語コメントで仕組みを解説。

## カバーするテーマ

| ファイル | テーマ | ひとことで言うと |
|---|---|---|
| `tokenizer.py` | トークナイズ | 文字 ⇄ 数字の翻訳機 |
| `transformer.py` | Transformer 実装・理解 | 注意機構を持つ超ミニGPTを自作 |
| `pretrain.py` | 事前学習 | 大量の文で「次の文字当て」を練習 |
| `lora.py` | LoRA | 巨大モデルを省メモリで改造する低ランク差分 |
| `sft.py` | SFT（教師ありFT） | お手本ペアを真似て"指示に従う"形を学ぶ |
| `dpo.py` | DPO（好み学習） | 「AよりB」を学び、基準モデルと比べて暴走を防ぐ |
| `prompting.py` | プロンプティング | 学習し直さず"聞き方"で出力を変える |
| `evaluate.py` | 評価 | perplexity と正解率で賢さを数値化 |

## セットアップ

```bash
arch -arm64 python3 -m venv .venv      # Apple Silicon
source .venv/bin/activate
pip install -r requirements.txt
```

## 動かす順番

```bash
export PYTHONPATH=src

python -m llm_finetune_lab.pretrain     # ① 事前学習 → checkpoints/pretrained.pt
python -m llm_finetune_lab.evaluate     # ② 賢さを測る（perplexity / 正解率）
python -m llm_finetune_lab.prompting    # ③ 聞き方の工夫を体感
python -m llm_finetune_lab.sft          # ④ LoRA で SFT → checkpoints/sft.pt
python -m llm_finetune_lab.dpo          # ⑤ LoRA で DPO → checkpoints/dpo.pt
```

> `pretrain` を最初に実行すること（他の工程はその成果物 `pretrained.pt` を土台にする）。

## テスト

```bash
pytest   # 10件 全green
```

## 結果（自前のミニ環境・GPU不要）
- **事前学習**: 損失 **3.04 → 0.11**（次の文字をほぼ当てられるように）→ 自然な文を生成。
- **LoRA**: 学習対象を全パラメータの **約7.28%** に削減（残りは凍結）。
- **評価**: perplexity **1.06** / 次文字 正解率 **97.5%**（小さい課題をほぼ完璧に予測）。
- **DPO**: 学習中に「悪い返答(rejected)の選ばれやすさ」が急降下＝好みを反映できた。

## 学習の流れ（おおまかな地図）

```
文章 ──[tokenizer]──▶ 数字列
                          │
                   [pretrain] 次の文字当てをひたすら練習（自己教師あり）
                          │  ＝「文章を続けられる」土台モデル
                          ▼
        ┌─────────────────┼─────────────────┐
   [SFT(+LoRA)]      [DPO(+LoRA)]        [prompting]
   お手本を真似て      good>bad を学び       学習せず
   指示に従わせる      好みを反映           聞き方で誘導
                          │
                          ▼
                     [evaluate] perplexity / 正解率で採点
```

## 設計メモ

- **decoder-only / causal**: 各位置は自分より前しか見られない（未来をカンニングしない）。
- **LoRA**: 元の重みは凍結し、低ランクの差分 `ΔW = B@A` だけを学習。学習パラメータが約7%に。
- **DPO**: policy と凍結 reference の log確率差を比較。β で reference から離れすぎを抑制。
- **再現性**: `torch.manual_seed` を各スクリプトで固定。

---
> 📝 学習目的の自己完結プロジェクトです（合成データで動く再実装）。AIコーディング支援を活用して実装し、設計・各処理を自分の言葉で説明できる状態にした上で公開しています。研究成果ではなく学習用です。
