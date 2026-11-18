"""
ex002/model.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験2（記憶制限手法 LQN, LNAQ, LMoQ および メモリレス手法 CG, MLQN, MLNAQ, MLMoQ）
向けモデル定義．

対象問題:
  - 1次元関数近似問題 f(x) = 1 + (x + 2x^2) sin(-x^2) (1-10-1, d = 31)
  - 5次元多峰性 Levy 関数 (5-50-1, d = 351)
"""

import torch
import torch.nn as nn


class FeedforwardMLP(nn.Module):
    """隠れ層1層（シグモイド活性化）・線形出力層を持つ階層型ニューラルネットワーク．"""

    def __init__(self, n_in: int = 1, n_hid: int = 10, n_out: int = 1, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.dtype = dtype
        self.fc1 = nn.Linear(n_in, n_hid, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True, dtype=dtype)

    @property
    def num_params(self) -> int:
        return self.n_in * self.n_hid + self.n_hid + self.n_hid * self.n_out + self.n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.sigmoid(self.fc1(x)))

    def get_flat_params(self) -> torch.Tensor:
        with torch.no_grad():
            w1 = self.fc1.weight.t().contiguous().view(-1)
            b1 = self.fc1.bias.view(-1)
            w2 = self.fc2.weight.t().contiguous().view(-1)
            b2 = self.fc2.bias.view(-1)
            return torch.cat([w1, b1, w2, b2], dim=0)

    def set_flat_params(self, flat_w: torch.Tensor) -> None:
        i1 = self.n_in * self.n_hid
        i2 = i1 + self.n_hid
        i3 = i2 + self.n_hid * self.n_out
        with torch.no_grad():
            self.fc1.weight.copy_(flat_w[:i1].view(self.n_in, self.n_hid).t())
            self.fc1.bias.copy_(flat_w[i1:i2])
            self.fc2.weight.copy_(flat_w[i2:i3].view(self.n_hid, self.n_out).t())
            self.fc2.bias.copy_(flat_w[i3:])


def half_mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """論文 式(15): E(w) = (1 / |Tr|) * sum (1/2) * ||d_p - o_p||^2."""
    diff = targets - outputs
    return 0.5 * torch.mean(torch.sum(diff * diff, dim=-1))


@torch.jit.script
def _jit_loss_only_ex2(
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
def _jit_loss_and_grad_ex2(
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
    """R 試行一括評価用の Flattened GEMM + TorchScript 高速モデルクラス．"""

    def __init__(self, n_in: int = 1, n_hid: int = 10, n_out: int = 1):
        assert n_out == 1
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid
        self.i2 = self.i1 + n_hid
        self.i3 = self.i2 + n_hid * n_out
        self.num_params = self.i3 + n_out

    def predict(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        R = w.shape[0]
        N = X.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid)
        o = torch.sum(h * W2.unsqueeze(0), dim=-1) + b2
        return o.t().unsqueeze(-1)

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D_col: torch.Tensor) -> torch.Tensor:
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_only_ex2(X, D_col, W1, b1, W2, b2, R, self.n_hid)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, XT: torch.Tensor, D_col: torch.Tensor):
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid)
        b2 = w[:, self.i3:].reshape(1, R)
        return _jit_loss_and_grad_ex2(X, XT, D_col, W1, b1, W2, b2, R, self.n_hid)
