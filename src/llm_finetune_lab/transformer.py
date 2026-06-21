"""Transformer（トランスフォーマー）をゼロから自作する。

これは ChatGPT 等の心臓部と「同じ仕組み」の超ミニ版。
やっていることは一つだけ:

    「ここまでの文字を見て、次の1文字を予想する」

これを大量に練習すると、文章を続けて書けるAIになる。

■ 全体の流れ（料理にたとえる）
  1. Embedding      … 文字番号を「意味ベクトル」に変える（材料を準備）
  2. Attention      … 各文字が「他のどの文字に注目すべきか」を計算（味見・調整）
  3. FeedForward    … 注目で集めた情報をこねる（加熱・調理）
  4. 2〜3を何層も重ねる（何度も味を深める）
  5. 最後に「次の文字の確率」を出す（盛り付け）

■ decoder-only / causal（コーザル＝因果）
  未来の文字をカンニングしないよう、各位置は「自分より前」しか見れない。
  これを causal mask（因果マスク）で実現する。テスト中に後ろの答えを
  隠すついたて、のイメージ。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """モデルの大きさ（ハイパーパラメータ）をまとめた設定箱。"""

    vocab_size: int          # 知っている文字の種類数
    block_size: int = 64     # 一度に見れる文脈の長さ（何文字さかのぼれるか）
    n_layer: int = 3         # Transformer ブロックを何段重ねるか
    n_head: int = 4          # アテンションの「視点」の数（マルチヘッド）
    n_embd: int = 64         # 意味ベクトルの次元（情報の太さ）
    dropout: float = 0.1     # 過学習よけ。学習中だけ一部をランダムに休ませる


class CausalSelfAttention(nn.Module):
    """自己注意機構（self-attention）。Transformer の主役。

    各文字が「文中の他のどの文字をどれだけ見るか(=注意の重み)」を自分で決め、
    重要な相手の情報を重み付きで集めてくる。

    たとえ話: 教室で「猫」という単語が、文中の「が」「走った」を見て
            『あ、自分は主語で、走る側だな』と理解する、その目配りの計算。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd は n_head で割り切れること"
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 入力から Query / Key / Value の3点セットを一気に作る全結合層。
        #   Query(クエリ) = 「私は何を探している？」という問い合わせ
        #   Key  (キー)   = 「私はこういう情報を持っているよ」という見出し
        #   Value(バリュー)= 「実際に渡す中身」
        # Q と K の相性が良い相手の V を多めにもらう、という仕組み。
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 集めてきた情報を最後に混ぜ直す層。
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果マスク: 下三角だけ1の行列。未来(右上)を見えなくする「ついたて」。
        # register_buffer = 学習しないが保存はしたい固定データ。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x の形: (B, T, C) = (バッチ数, 文の長さ, 意味ベクトルの次元)
        B, T, C = x.shape

        # Q, K, V を作って3つに分ける。
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # マルチヘッド: 1つの大きな注意を、小さな「視点」h個に分割する。
        # 視点ごとに別の観点（文法・意味・位置…）で注目できる＝多面的に見る。
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # 注意スコア = Q と K の内積。相性の良さを全ペアで計算する。
        # √head_dim で割るのは、値が大きくなりすぎて学習が不安定になるのを防ぐため。
        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, h, T, T)

        # 未来をマスク: ついたての外(=未来)を -∞ にして、softmax で 0 にする。
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # softmax で「合計1の注意の重み」に変換（誰をどれだけ見るかの配分）。
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 重み付きで V を集める = 注目した相手の中身を取り込む。
        y = att @ v  # (B, h, T, d)

        # 視点を1本に戻して混ぜる。
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    """フィードフォワード層（FeedForward / MLP）。

    アテンションで「集めた」情報を、各文字ごとに非線形でこね回して
    特徴を深める。いったん4倍に広げてから戻す（広い作業台で調理する）。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU = なめらかな活性化関数。負の値をやんわり通す（ReLUの上品版）。
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer ブロック1段 = アテンション + MLP。

    ポイントは「残差接続(residual)」: x = x + f(x)。
    元の x を捨てずに、変化分だけを足す。情報が深い層でも消えにくくなる
    （来た道のメモを残したまま付け足していくイメージ）。
    LayerNorm は各層の入力の大きさを整える「整流装置」。
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # 注目して情報を集める
        x = x + self.mlp(self.ln2(x))    # 集めた情報をこねる
        return x


class MiniGPT(nn.Module):
    """超ミニ GPT 本体。文字列を受け取り「次の文字の確率」を出す。"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # 文字番号 → 意味ベクトル（単語の意味の表）。
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置番号 → 位置ベクトル（何文字目かを教える表）。
        # アテンションは順番を知らないので、位置情報を足して教える必要がある。
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer ブロックを n_layer 段重ねる。
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 最後に「意味ベクトル → 各文字のスコア(logits)」へ変換する出口。
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx の形: (B, T) = 文字番号の列。
        B, T = idx.shape
        assert T <= self.config.block_size, "文脈長 block_size を超えた"

        pos = torch.arange(T, device=idx.device)

        # 「意味」と「位置」を足し合わせて、各文字の初期ベクトルを作る。
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.head(x)  # (B, T, vocab_size) 各位置で次の文字のスコア

        loss = None
        if targets is not None:
            # 学習時: 予測(logits)と正解(targets)のズレを cross entropy で測る。
            # これが小さいほど「次の文字当て」が上手＝損失(loss)。
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """文章生成: 次の1文字を予想 → くっつける、を繰り返す（自己回帰）。

        temperature(温度): 大きいほど大胆・ランダム、小さいほど無難・堅実。
        top_k: 上位k個の候補だけから選ぶ（変な文字が出る事故を防ぐ）。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 文脈が長すぎたら、見れる範囲(block_size)に末尾を切り詰める。
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 一番最後の位置の予測だけ使う

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # 確率に従って1文字サンプリング
            idx = torch.cat([idx, next_id], dim=1)
        return idx
