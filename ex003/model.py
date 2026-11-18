"""
ex003/model.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験3（分類問題ベンチマーク: Three-Spirals, 8x8 MNIST, 28x28 MNIST風分類）
向けモデル定義．

論文 2.2節 式(19)-(21) に従い:
  - 隠れ層活性化関数: シグモイド関数 (式(16))
  - 出力層活性化関数: ソフトマックス関数 (式(20))
  - 誤差関数: 交差エントロピー誤差 CE (式(19)): E_p(w) = -sum_i d_{i,p} log(x_{i,p}^{out})
  - 解析的勾配 (式(21)): dE_p/dz_{i,p}^{out} = x_{i,p}^{out} - d_{i,p}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardClassifier(nn.Module):
    """隠れ層1層（シグモイド）＋出力層（線形ロジット -> Softmax）の多クラス分類 MLP．"""

    def __init__(self, n_in: int, n_hid: int, n_out: int, dtype: torch.dtype = torch.float64):
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
        """ロジットを返す (数値安定な log_softmax / cross_entropy 用)．"""
        h = torch.sigmoid(self.fc1(x))
        return self.fc2(h)

    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)

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


def cross_entropy_onehot_loss(logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
    """論文 式(19): E(w) = -(1 / |Tr|) * sum_{p, i} d_{i,p} log(softmax(z_p)_i)."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.mean(torch.sum(targets_onehot * log_probs, dim=-1))


class BatchedClassifier:
    """
    R 個の独立試行を単一テンソル (R, D_w) で保持し，GPU 上で一括並列に
    Softmax + 交差エントロピー誤差 (式(19), (20)) および解析的誤差逆伝播 (式(21)) を計算するクラス．
    """

    def __init__(self, n_in: int, n_hid: int, n_out: int):
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.i1 = n_in * n_hid
        self.i2 = self.i1 + n_hid
        self.i3 = self.i2 + n_hid * n_out
        self.num_params = self.i3 + n_out

    def _unpack(self, w: torch.Tensor):
        R = w.shape[0]
        W1 = w[:, :self.i1].reshape(R, self.n_in, self.n_hid).permute(1, 0, 2).reshape(self.n_in, R * self.n_hid)
        b1 = w[:, self.i1:self.i2].reshape(1, R * self.n_hid)
        W2 = w[:, self.i2:self.i3].reshape(R, self.n_hid, self.n_out)
        b2 = w[:, self.i3:].reshape(R, 1, self.n_out)
        return R, W1, b1, W2, b2

    def predict_probs(self, w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """予測確率 (R, N, L) を返す．"""
        R, W1, b1, W2, b2 = self._unpack(w)
        N = X.shape[0]
        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid).permute(1, 0, 2)  # (R, N, H)
        logits = torch.bmm(h, W2) + b2  # (R, N, L)
        return torch.softmax(logits, dim=-1)

    def accuracy(self, w: torch.Tensor, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """分類精度 (%) (R,) を返す．labels は整数ラベル (N,)．"""
        probs = self.predict_probs(w, X)
        preds = torch.argmax(probs, dim=-1)  # (R, N)
        return (preds == labels.unsqueeze(0)).to(torch.float64).mean(dim=1) * 100.0

    def loss_only(self, w: torch.Tensor, X: torch.Tensor, D_onehot: torch.Tensor) -> torch.Tensor:
        """交差エントロピー誤差 E(w) のみ (R,) を計算する．D_onehot は (N, L)．"""
        R, W1, b1, W2, b2 = self._unpack(w)
        N = X.shape[0]
        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid).permute(1, 0, 2)  # (R, N, H)
        logits = torch.bmm(h, W2) + b2  # (R, N, L)
        log_probs = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(D_onehot.unsqueeze(0) * log_probs, dim=-1), dim=1)

    def loss_and_grad(self, w: torch.Tensor, X: torch.Tensor, XT: torch.Tensor, D_onehot: torch.Tensor):
        """
        交差エントロピー誤差 E(w) (R,) と解析的勾配 (R, D_w) を計算する．
        式(21): dE/dz_out = (softmax(z_out) - D_onehot) / N
        """
        R, W1, b1, W2, b2 = self._unpack(w)
        N = X.shape[0]
        h = torch.sigmoid(torch.mm(X, W1) + b1).view(N, R, self.n_hid).permute(1, 0, 2)  # (R, N, H)
        logits = torch.bmm(h, W2) + b2  # (R, N, L)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        loss = -torch.mean(torch.sum(D_onehot.unsqueeze(0) * log_probs, dim=-1), dim=1)

        d_z2 = (probs - D_onehot.unsqueeze(0)) * (1.0 / float(N))  # (R, N, L)
        g_W2 = torch.bmm(h.transpose(1, 2), d_z2).reshape(R, self.n_hid * self.n_out)
        g_b2 = torch.sum(d_z2, dim=1)  # (R, L)

        d_h = torch.bmm(d_z2, W2.transpose(1, 2))  # (R, N, H)
        d_z1 = (d_h * h * (1.0 - h)).permute(1, 0, 2).reshape(N, R * self.n_hid)  # (N, R*H)
        g_b1 = torch.sum(d_z1.view(N, R, self.n_hid), dim=0)  # (R, H)
        g_W1_raw = torch.mm(XT, d_z1)  # (n_in, R*H)
        g_W1 = g_W1_raw.view(self.n_in, R, self.n_hid).permute(1, 0, 2).reshape(R, self.i1)

        grad = torch.cat([g_W1, g_b1, g_W2, g_b2], dim=1)
        return loss, grad
