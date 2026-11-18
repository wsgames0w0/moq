"""
ex001/model.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験1（フルサイズ準ニュートン法群: QN, MSQN, NAQ, AdaNAQ, MoQ, AdaMoQ）
向けモデル定義．

1. FeedforwardMLP (torch.nn.Module):
   標準的なPyTorchモジュールとしての階層型ニューラルネットワーク（シグモイド活性化＋線形出力）．
   カスタム torch.optim.Optimizer (MoQFamilyOptimizer 等) と組み合わせて使用する．
2. BatchedMLP:
   独立した R 回の試行を単一テンソル (R, D_w) としてGPU上で一括並列計算するための高速モデルクラス．
   Flattened GEMM と TorchScript (@torch.jit.script) による要素演算融合を備える．
"""

import torch
import torch.nn as nn


class FeedforwardMLP(nn.Module):
    """隠れ層1層（シグモイド活性化）・線形出力層を持つフィードフォワードニューラルネットワーク．"""

    def __init__(self, n_in: int = 2, n_hid: int = 55, n_out: int = 1, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.dtype = dtype

        self.fc1 = nn.Linear(n_in, n_hid, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True, dtype=dtype)

    @property
    def num_params(self) -> int:
        """全パラメータ数 dim(w) を返す．"""
        return self.n_in * self.n_hid + self.n_hid + self.n_hid * self.n_out + self.n_out

    def reset_parameters_uniform(self, low: float = -0.5, high: float = 0.5) -> None:
        """論文に従い，全重み・バイアスを一様乱数 [low, high] で初期化する．"""
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(low, high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播計算: h = sig(W1 x + b1), o = W2 h + b2."""
        h = torch.sigmoid(self.fc1(x))
        o = self.fc2(h)
        return o

    def get_flat_params(self) -> torch.Tensor:
        """BatchedMLP と同一の並び順 [W1(n_in, n_hid), b1(n_hid), W2(n_hid, n_out), b2(n_out)] で1次元ベクトルを取得する．"""
        with torch.no_grad():
            w1 = self.fc1.weight.t().contiguous().view(-1)
            b1 = self.fc1.bias.view(-1)
            w2 = self.fc2.weight.t().contiguous().view(-1)
            b2 = self.fc2.bias.view(-1)
            return torch.cat([w1, b1, w2, b2], dim=0)

    def set_flat_params(self, flat_w: torch.Tensor) -> None:
        """1次元パラメータベクトル flat_w を各層の重み・バイアスへ書き戻す．"""
        i1 = self.n_in * self.n_hid
        i2 = i1 + self.n_hid
        i3 = i2 + self.n_hid * self.n_out
        with torch.no_grad():
            self.fc1.weight.copy_(flat_w[:i1].view(self.n_in, self.n_hid).t())
            self.fc1.bias.copy_(flat_w[i1:i2])
            self.fc2.weight.copy_(flat_w[i2:i3].view(self.n_hid, self.n_out).t())
            self.fc2.bias.copy_(flat_w[i3:])


def half_mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    論文 式(5), (15) の誤差関数 E(w) = (1 / |Tr|) * sum_{p in Tr} (1/2) * ||d_p - o_p||^2 を計算する．
    """
    diff = targets - outputs
    return 0.5 * torch.mean(torch.sum(diff * diff, dim=-1))


@torch.jit.script
def _jit_loss_only_1out(
    X: torch.Tensor,
    D_col: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    R: int,
    H: int,
) -> torch.Tensor:
    N = X.shape[0]
    h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, H)
    o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
    diff = o - D_col
    return 0.5 * torch.mean(diff * diff, dim=0)


@torch.jit.script
def _jit_loss_and_grad_1out(
    X: torch.Tensor,
    XT: torch.Tensor,
    D_col: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    R: int,
    H: int,
):
    N = X.shape[0]
    h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, H)
    o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
    diff = o - D_col
    loss = 0.5 * torch.mean(diff * diff, dim=0)

    d_o = diff * (1.0 / float(N))
    g_W2 = torch.sum(h * d_o.unsqueeze(-1), dim=0)
    g_b2 = torch.sum(d_o, dim=0).unsqueeze(-1)

    d_z1 = (d_o.unsqueeze(-1) * W2.unsqueeze(0)) * (h * (1.0 - h))
    g_b1 = torch.sum(d_z1, dim=0)
    g_W1_raw = torch.mm(XT, d_z1.view(N, R * H))
    g_W1 = g_W1_raw.view(-1, R, H).permute(1, 0, 2).reshape(R, -1)

    return loss, torch.cat([g_W1, g_b1, g_W2, g_b2], dim=1)


class BatchedMLP:
    """
    R 個の独立試行を GPU 上で一括並列評価・勾配計算する高速モデルクラス．
    Flattened GEMM (全試行の第1層重みを横連結した1回の torch.mm) と TorchScript 融合により
    小規模〜中規模ニューラルネットワークの大規模モンテカルロ試行を高速実行する．
    """

    def __init__(self, n_in: int = 2, n_hid: int = 55, n_out: int = 1):
        assert n_out == 1, "BatchedMLP currently assumes scalar output (n_out=1)"
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid
        self.i2 = self.i1 + n_hid
        self.i3 = self.i2 + n_hid * n_out
        self.num_params = self.i3 + n_out

    def predict(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """順伝播出力 o (R, N, 1) を返す．"""
        R = w.shape[0]
        N = X.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid)
        o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2  # (N, R)
        return o.t().unsqueeze(-1)

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D_col: torch.Tensor) -> torch.Tensor:
        """誤差関数 E(w) のみ (R,) を高速計算する（先読み点・Armijo直線探索の候補評価用）．"""
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_only_1out(X, D_col, W1, b1, W2, b2, R, self.n_hid)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, XT: torch.Tensor, D_col: torch.Tensor):
        """誤差関数 E(w) (R,) と解析的勾配ベクトル grad E(w) (R, D_w) を一括計算する．"""
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_and_grad_1out(X, XT, D_col, W1, b1, W2, b2, R, self.n_hid)
