"""
ex002/utils.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験2 (第5章: 記憶制限手法, 第6章: メモリレス手法) 用アルゴリズム実装．

実装内容:
1. 理論計算コスト・メモリ量の算出 (表6, 表13, 表17)
2. PyTorch 標準 torch.optim.Optimizer 継承クラス:
   - LimitedMemoryMoQOptimizer (LQN, LNAQ, LMoQ: Two-Loop Recursion)
   - MemoryLessMoQOptimizer    (CG-HS, MLQN, MLNAQ, MLMoQ: O(d) 内積更新)
3. 1次元関数近似 (式(124), |Tr|=400, 1-10-1, d=31) および
   5次元 Levy 関数 (式(102), |Tr|=5000, 5-50-1, d=351) 用データセット生成・読込
4. GPU バッチ並列学習エンジン (Two-Loop Recursion バッチ版 & MemoryLess バッチ版)
"""

import math
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig")
    os.makedirs(_mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = _mpl_dir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import Optimizer
from tqdm import tqdm

from model import BatchedMLP


def setup_japanese_matplotlib() -> None:
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def adaptive_momentum_step_scalar(theta: float, gamma: float = 1e-5) -> Tuple[float, float]:
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + math.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def adaptive_momentum_step_tensor(
    theta: torch.Tensor, gamma: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + torch.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def compute_theoretical_cost_and_storage(
    d_values: np.ndarray,
    n: int = 1000,
    zeta: int = 1,
    m_list: Tuple[int, ...] = (1, 8, 32, 128, 512),
) -> pd.DataFrame:
    """
    論文 表6, 表13, 表17 に基づき各手法の1反復あたりの計算コストと必要メモリ量を算出する．
    """
    rows = []
    for d in d_values:
        d = int(d)
        row: Dict[str, float] = {"d": d}
        # Normal
        row["QN_cost"] = n * d + d * d + zeta * n * d
        row["NAQ_cost"] = 2 * n * d + d * d + zeta * n * d
        row["MoQ_cost"] = n * d + d * d + zeta * n * d
        row["QN_mem"] = d * d + 2 * (n + 1) * d
        row["NAQ_mem"] = d * d + 2 * (n + 1) * d
        row["MoQ_mem"] = d * d + (3 * n + 2) * d
        # Limited-Memory
        for m in m_list:
            row[f"LQN_m{m}_cost"] = n * d + 4 * m * d + 2 * d + zeta * n * d
            row[f"LNAQ_m{m}_cost"] = 2 * n * d + 4 * m * d + 2 * d + zeta * n * d
            row[f"LMoQ_m{m}_cost"] = n * d + 4 * m * d + 2 * d + zeta * n * d
            row[f"LQN_m{m}_mem"] = 2 * m * d + 2 * (n + 1) * d
            row[f"LMoQ_m{m}_mem"] = 2 * m * d + (3 * n + 2) * d
        # MemoryLess
        row["MLQN_cost"] = n * d + 4 * d + zeta * n * d
        row["MLNAQ_cost"] = 2 * n * d + 4 * d + zeta * n * d
        row["MLMoQ_cost"] = n * d + 4 * d + zeta * n * d
        row["MLQN_mem"] = 2 * (n + 1) * d
        row["MLMoQ_mem"] = (3 * n + 2) * d
        rows.append(row)
    return pd.DataFrame(rows)


def _global_conv_q_scalar(
    delta: torch.Tensor, p_k: torch.Tensor, g_lin: torch.Tensor, use_gc: bool = True, gc_scale: float = 1.0
) -> torch.Tensor:
    if not use_gc:
        return delta
    p_norm_sq = float(torch.dot(p_k, p_k).item())
    if p_norm_sq <= 1e-30:
        return delta
    g_norm_sq = float(torch.dot(g_lin, g_lin).item())
    g_norm = math.sqrt(g_norm_sq)
    omega = (2.0 if g_norm_sq > 1e-2 else 100.0) * gc_scale
    dp = float(torch.dot(delta, p_k).item())
    xi_k = omega * g_norm + max(-dp / p_norm_sq, 0.0)
    return delta + xi_k * p_k


class LimitedMemoryMoQOptimizer(Optimizer):
    """
    論文 第5章: 記憶制限準ニュートン法群 (LQN, LNAQ, LMoQ) の汎用 PyTorch Optimizer 実装．
    Two-Loop Recursion (Algorithms 14, 16, 18) により直近 m 個のベクトル対のみを用いて
    O(m d) の空間・時間計算量で探索方向ベクトル g_k を算出する．
    """

    VALID_ALGOS = ("LQN", "LNAQ", "LMoQ")

    def __init__(
        self,
        params,
        algo: str = "LMoQ",
        memory_size: int = 8,
        use_adaptive_mu: bool = True,
        mu: float = 0.85,
        gamma: float = 1e-5,
        chi: float = 1e-3,
        tau: float = 0.5,
        ls_max: int = 10,
        max_step_norm: float = 20.0,
        curvature_eps: float = 1e-14,
        use_global_conv: bool = False,
        global_conv_scale: float = 1.0,
        max_mu: float = 0.92,
    ):
        if algo not in self.VALID_ALGOS:
            raise ValueError(f"Unsupported algo '{algo}'")
        defaults = dict(
            algo=algo,
            memory_size=memory_size,
            use_adaptive_mu=use_adaptive_mu,
            mu=mu,
            gamma=gamma,
            chi=chi,
            tau=tau,
            ls_max=ls_max,
            max_step_norm=max_step_norm,
            curvature_eps=curvature_eps,
            use_global_conv=use_global_conv,
            global_conv_scale=global_conv_scale,
            max_mu=max_mu,
        )
        super().__init__(params, defaults)
        self._params_list: List[torch.nn.Parameter] = [
            p for group in self.param_groups for p in group["params"]
        ]

    def _gather_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self._params_list], dim=0)

    def _set_flat_params(self, flat_w: torch.Tensor) -> None:
        offset = 0
        for p in self._params_list:
            numel = p.numel()
            p.data.copy_(flat_w[offset : offset + numel].view_as(p.data))
            offset += numel

    def _gather_flat_grad(self) -> torch.Tensor:
        return torch.cat(
            [(torch.zeros_like(p.data) if p.grad is None else p.grad.data).view(-1) for p in self._params_list],
            dim=0,
        )

    @staticmethod
    def two_loop_direction(
        grad_base: torch.Tensor,
        p_hist: Deque[torch.Tensor],
        q_hist: Deque[torch.Tensor],
        pq_hist: Deque[float],
    ) -> torch.Tensor:
        """
        Algorithms 14, 16, 18 の Two-Loop Recursion:
          g = -grad_base から開始し，直近 m ステップの (p_i, q_i) で逆ヘッセ積を近似する．
        """
        g = -grad_base.clone()
        n_hist = len(p_hist)
        if n_hist == 0:
            return g

        alphas = [0.0] * n_hist
        # Loop 1: 新しい履歴 -> 古い履歴
        for i in range(n_hist - 1, -1, -1):
            phi_i = float(torch.dot(p_hist[i], g).item()) / pq_hist[i]
            alphas[i] = phi_i
            g.add_(q_hist[i], alpha=-phi_i)

        # 中央スケーリング: gamma_k = (p_last^T q_last) / (q_last^T q_last)
        qq_last = float(torch.dot(q_hist[-1], q_hist[-1]).item())
        if qq_last > 1e-30:
            gamma_k = pq_hist[-1] / qq_last
            g.mul_(gamma_k)

        # Loop 2: 古い履歴 -> 新しい履歴
        for i in range(n_hist):
            tau = float(torch.dot(q_hist[i], g).item()) / pq_hist[i]
            g.add_(p_hist[i], alpha=(alphas[i] - tau))

        return g

    @torch.no_grad()
    def step(self, closure: Callable[[bool], torch.Tensor]) -> Tuple[float, float, float, float]:
        cfg = self.param_groups[0]
        algo: str = cfg["algo"]
        m_size: int = cfg["memory_size"]
        use_ada: bool = cfg["use_adaptive_mu"]
        mu_fixed: float = cfg["mu"]
        gamma: float = cfg["gamma"]
        chi: float = cfg["chi"]
        tau: float = cfg["tau"]
        ls_max: int = cfg["ls_max"]
        max_step_norm: float = cfg["max_step_norm"]
        curv_eps: float = cfg["curvature_eps"]
        use_gc: bool = cfg["use_global_conv"]
        gc_scale: float = cfg["global_conv_scale"]
        max_mu: float = cfg["max_mu"]

        w_k = self._gather_flat_params()
        state = self.state["global_lmoq_state"]
        if len(state) == 0:
            state["step"] = 0
            state["theta"] = 1.0
            state["v"] = torch.zeros_like(w_k)
            state["p_hist"] = deque(maxlen=m_size)
            state["q_hist"] = deque(maxlen=m_size)
            state["pq_hist"] = deque(maxlen=m_size)
            with torch.enable_grad():
                loss_init = closure(True)
            g_init = self._gather_flat_grad().clone()
            state["loss_k"] = float(loss_init.item())
            state["grad_k"] = g_init
            state["grad_prev"] = g_init.clone()

        state["step"] += 1
        k: int = state["step"]
        v_k: torch.Tensor = state["v"]
        loss_k: float = state["loss_k"]
        grad_k: torch.Tensor = state["grad_k"]
        grad_prev: torch.Tensor = state["grad_prev"]
        p_hist: Deque[torch.Tensor] = state["p_hist"]
        q_hist: Deque[torch.Tensor] = state["q_hist"]
        pq_hist: Deque[float] = state["pq_hist"]

        if algo == "LQN":
            mu_k = 0.0
        elif use_ada:
            mu_k, theta_next = adaptive_momentum_step_scalar(state["theta"], gamma=gamma)
            mu_k = min(mu_k, max_mu)
            state["theta"] = theta_next
        else:
            mu_k = mu_fixed

        if algo == "LQN":
            w_base, loss_base, grad_base = w_k, loss_k, grad_k
        elif algo == "LNAQ":
            if mu_k == 0.0:
                w_base, loss_base, grad_base = w_k, loss_k, grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                with torch.enable_grad():
                    loss_base_t = closure(True)
                loss_base = float(loss_base_t.item())
                grad_base = self._gather_flat_grad().clone()
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if use_ada:
                        state["theta"] = 1.0
                    w_base, loss_base, grad_base = w_k, loss_k, grad_k
        else:  # LMoQ
            if mu_k == 0.0 or k == 1:
                w_base, loss_base, grad_base = w_k, loss_k, grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                loss_base = float(closure(False).item())
                grad_base = (1.0 + mu_k) * grad_k - mu_k * grad_prev
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if use_ada:
                        state["theta"] = 1.0
                    w_base, loss_base, grad_base = w_k, loss_k, grad_k

        g_raw = self.two_loop_direction(grad_base, p_hist, q_hist, pq_hist)
        scale = max(1.0, float(torch.norm(g_raw).item()) / max_step_norm)
        g_k = g_raw / scale
        slope = float(torch.dot(grad_base, g_k).item())
        if slope >= 0.0:
            p_hist.clear()
            q_hist.clear()
            pq_hist.clear()
            g_k = -grad_base / max(1.0, float(torch.norm(grad_base).item()) / max_step_norm)
            slope = float(torch.dot(grad_base, g_k).item())

        alpha = tau ** (ls_max - 1)
        for m in range(ls_max):
            ca = tau ** m
            self._set_flat_params(w_base + ca * g_k)
            if float(closure(False).item()) <= loss_base + chi * ca * slope:
                alpha = ca
                break

        v_next = mu_k * v_k + alpha * g_k
        w_next = w_k + v_next
        self._set_flat_params(w_next)

        with torch.enable_grad():
            loss_next_t = closure(True)
        loss_next = float(loss_next_t.item())
        grad_next = self._gather_flat_grad().clone()

        p_k = alpha * g_k
        delta = grad_next - grad_base
        q_k = _global_conv_q_scalar(delta, p_k, grad_base, use_gc=use_gc, gc_scale=gc_scale)
        pq = float(torch.dot(p_k, q_k).item())
        if pq > curv_eps:
            p_hist.append(p_k)
            q_hist.append(q_k)
            pq_hist.append(pq)
        else:
            p_hist.clear()
            q_hist.clear()
            pq_hist.clear()

        state["grad_prev"] = grad_k
        state["v"] = v_next
        state["loss_k"] = loss_next
        state["grad_k"] = grad_next
        return loss_next, float(torch.norm(grad_next).item()), alpha, mu_k


class MemoryLessMoQOptimizer(Optimizer):
    """
    論文 第6章: メモリレス準ニュートン法群 (CG-HS, MLQN, MLNAQ, MLMoQ) の汎用 PyTorch Optimizer 実装．
    逆ヘッセ行列や過去の履歴キューを一切保持せず，直前 1 ステップの (p_{k-1}, q_{k-1}) との
    内積スカラー (beta_k, psi_k, omega_k, tau_k) のみで O(d) の計算量で探索方向を求める．
    """

    VALID_ALGOS = ("CG", "MLQN", "MLNAQ", "MLMoQ")

    def __init__(
        self,
        params,
        algo: str = "MLMoQ",
        use_adaptive_mu: bool = True,
        mu: float = 0.85,
        gamma: float = 1e-5,
        chi: float = 1e-3,
        tau: float = 0.5,
        ls_max: int = 10,
        max_step_norm: float = 20.0,
        curvature_eps: float = 1e-14,
        use_global_conv: bool = False,
        global_conv_scale: float = 1.0,
        max_mu: float = 0.92,
    ):
        if algo not in self.VALID_ALGOS:
            raise ValueError(f"Unsupported algo '{algo}'")
        defaults = dict(
            algo=algo,
            use_adaptive_mu=use_adaptive_mu,
            mu=mu,
            gamma=gamma,
            chi=chi,
            tau=tau,
            ls_max=ls_max,
            max_step_norm=max_step_norm,
            curvature_eps=curvature_eps,
            use_global_conv=use_global_conv,
            global_conv_scale=global_conv_scale,
            max_mu=max_mu,
        )
        super().__init__(params, defaults)
        self._params_list: List[torch.nn.Parameter] = [
            p for group in self.param_groups for p in group["params"]
        ]

    def _gather_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self._params_list], dim=0)

    def _set_flat_params(self, flat_w: torch.Tensor) -> None:
        offset = 0
        for p in self._params_list:
            numel = p.numel()
            p.data.copy_(flat_w[offset : offset + numel].view_as(p.data))
            offset += numel

    def _gather_flat_grad(self) -> torch.Tensor:
        return torch.cat(
            [(torch.zeros_like(p.data) if p.grad is None else p.grad.data).view(-1) for p in self._params_list],
            dim=0,
        )

    @staticmethod
    def memoryless_direction(
        grad_base: torch.Tensor,
        p_prev: torch.Tensor,
        q_prev: torch.Tensor,
        pq_prev: float,
        qq_prev: float,
    ) -> torch.Tensor:
        """
        式(135), (141), (146):
          g_k = -tau_k * (grad_base - (beta_k * q + psi_k * p)) - (1 + tau_k * omega_k) * beta_k * p
        """
        if pq_prev <= 1e-14 or qq_prev <= 1e-30:
            return -grad_base.clone()
        beta_k = float(torch.dot(p_prev, grad_base).item()) / pq_prev
        psi_k = float(torch.dot(q_prev, grad_base).item()) / pq_prev
        omega_k = qq_prev / pq_prev
        tau_k = pq_prev / qq_prev
        g = -tau_k * (grad_base - (beta_k * q_prev + psi_k * p_prev)) - (1.0 + tau_k * omega_k) * beta_k * p_prev
        return g

    @torch.no_grad()
    def step(self, closure: Callable[[bool], torch.Tensor]) -> Tuple[float, float, float, float]:
        cfg = self.param_groups[0]
        algo: str = cfg["algo"]
        use_ada: bool = cfg["use_adaptive_mu"]
        mu_fixed: float = cfg["mu"]
        gamma: float = cfg["gamma"]
        chi: float = cfg["chi"]
        tau: float = cfg["tau"]
        ls_max: int = cfg["ls_max"]
        max_step_norm: float = cfg["max_step_norm"]
        curv_eps: float = cfg["curvature_eps"]
        use_gc: bool = cfg["use_global_conv"]
        gc_scale: float = cfg["global_conv_scale"]
        max_mu: float = cfg["max_mu"]

        w_k = self._gather_flat_params()
        state = self.state["global_mlmoq_state"]
        if len(state) == 0:
            state["step"] = 0
            state["theta"] = 1.0
            state["v"] = torch.zeros_like(w_k)
            state["has_prev"] = False
            state["p_prev"] = torch.zeros_like(w_k)
            state["q_prev"] = torch.zeros_like(w_k)
            state["pq_prev"] = 0.0
            state["qq_prev"] = 0.0
            state["g_cg_prev"] = torch.zeros_like(w_k)
            state["y_cg_prev"] = torch.zeros_like(w_k)
            with torch.enable_grad():
                loss_init = closure(True)
            g_init = self._gather_flat_grad().clone()
            state["loss_k"] = float(loss_init.item())
            state["grad_k"] = g_init
            state["grad_prev"] = g_init.clone()

        state["step"] += 1
        k: int = state["step"]
        v_k: torch.Tensor = state["v"]
        loss_k: float = state["loss_k"]
        grad_k: torch.Tensor = state["grad_k"]
        grad_prev: torch.Tensor = state["grad_prev"]

        if algo in ("CG", "MLQN"):
            mu_k = 0.0
        elif use_ada:
            mu_k, theta_next = adaptive_momentum_step_scalar(state["theta"], gamma=gamma)
            mu_k = min(mu_k, max_mu)
            state["theta"] = theta_next
        else:
            mu_k = mu_fixed

        if algo in ("CG", "MLQN"):
            w_base, loss_base, grad_base = w_k, loss_k, grad_k
        elif algo == "MLNAQ":
            if mu_k == 0.0:
                w_base, loss_base, grad_base = w_k, loss_k, grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                with torch.enable_grad():
                    loss_base_t = closure(True)
                loss_base = float(loss_base_t.item())
                grad_base = self._gather_flat_grad().clone()
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if use_ada:
                        state["theta"] = 1.0
                    w_base, loss_base, grad_base = w_k, loss_k, grad_k
        else:  # MLMoQ
            if mu_k == 0.0 or k == 1:
                w_base, loss_base, grad_base = w_k, loss_k, grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                loss_base = float(closure(False).item())
                grad_base = (1.0 + mu_k) * grad_k - mu_k * grad_prev
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if use_ada:
                        state["theta"] = 1.0
                    w_base, loss_base, grad_base = w_k, loss_k, grad_k

        if algo == "CG":
            if k == 1 or not state["has_prev"]:
                g_raw = -grad_base.clone()
            else:
                gy = float(torch.dot(grad_base, state["y_cg_prev"]).item())
                denom = float(torch.dot(state["g_cg_prev"], state["y_cg_prev"]).item())
                beta_hs = (gy / denom) if abs(denom) > 1e-20 else 0.0
                beta_hs = max(0.0, beta_hs)  # 非降下防止のための安全リセット
                g_raw = -grad_base + beta_hs * state["g_cg_prev"]
        elif state["has_prev"]:
            g_raw = self.memoryless_direction(
                grad_base, state["p_prev"], state["q_prev"], state["pq_prev"], state["qq_prev"]
            )
        else:
            g_raw = -grad_base.clone()

        g_k = g_raw / max(1.0, float(torch.norm(g_raw).item()) / max_step_norm)
        slope = float(torch.dot(grad_base, g_k).item())
        if slope >= 0.0:
            state["has_prev"] = False
            g_k = -grad_base / max(1.0, float(torch.norm(grad_base).item()) / max_step_norm)
            slope = float(torch.dot(grad_base, g_k).item())

        alpha = tau ** (ls_max - 1)
        for m in range(ls_max):
            ca = tau ** m
            self._set_flat_params(w_base + ca * g_k)
            if float(closure(False).item()) <= loss_base + chi * ca * slope:
                alpha = ca
                break

        v_next = mu_k * v_k + alpha * g_k
        w_next = w_k + v_next
        self._set_flat_params(w_next)

        with torch.enable_grad():
            loss_next_t = closure(True)
        loss_next = float(loss_next_t.item())
        grad_next = self._gather_flat_grad().clone()

        if algo == "CG":
            state["g_cg_prev"] = g_k.clone()
            state["y_cg_prev"] = grad_next - grad_k
            state["has_prev"] = True
        else:
            p_k = alpha * g_k
            delta = grad_next - grad_base
            q_k = _global_conv_q_scalar(delta, p_k, grad_base, use_gc=use_gc, gc_scale=gc_scale)
            pq = float(torch.dot(p_k, q_k).item())
            qq = float(torch.dot(q_k, q_k).item())
            if pq > curv_eps and qq > 1e-30:
                state["p_prev"] = p_k
                state["q_prev"] = q_k
                state["pq_prev"] = pq
                state["qq_prev"] = qq
                state["has_prev"] = True
            else:
                state["has_prev"] = False

        state["grad_prev"] = grad_k
        state["v"] = v_next
        state["loss_k"] = loss_next
        state["grad_k"] = grad_next
        return loss_next, float(torch.norm(grad_next).item()), alpha, mu_k


# =====================================================================
# データセット生成・読込 (1D f(x) 式(124) & 5D Levy 式(102))
# =====================================================================

def target_function_1d(x: np.ndarray) -> np.ndarray:
    """論文 式(124): f(x) = 1 + (x + 2x^2) sin(-x^2), |x| <= 4."""
    return 1.0 + (x + 2.0 * (x ** 2)) * np.sin(-(x ** 2))


def load_or_create_1d_f1_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    実験2A用データセット (論文 5.4.1節, 6.5.1節):
      訓練データ |Tr| = 400 (x in [-4, 4), 0.02刻み), テストデータ |Te| = 10,000 (一様乱数)
      [-1.0, 1.0] 線形正規化．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_ex2_f1_1d.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        x_tr_raw, y_tr_raw = d["x_tr_raw"], d["y_tr_raw"]
        x_te_raw, y_te_raw = d["x_te_raw"], d["y_te_raw"]
    else:
        rng = np.random.default_rng(seed)
        x_tr_raw = np.arange(-4.0, 4.0, 0.02, dtype=np.float64)
        y_tr_raw = target_function_1d(x_tr_raw)
        x_te_raw = np.sort(rng.uniform(-4.0, 4.0, size=10000).astype(np.float64))
        y_te_raw = target_function_1d(x_te_raw)
        np.savez(npz_path, x_tr_raw=x_tr_raw, y_tr_raw=y_tr_raw, x_te_raw=x_te_raw, y_te_raw=y_te_raw)

    x_min, x_max = -4.0, 4.0
    y_min = min(float(y_tr_raw.min()), float(y_te_raw.min()))
    y_max = max(float(y_tr_raw.max()), float(y_te_raw.max()))

    x_tr_n = 2.0 * (x_tr_raw - x_min) / (x_max - x_min) - 1.0
    y_tr_n = 2.0 * (y_tr_raw - y_min) / (y_max - y_min) - 1.0
    x_te_n = 2.0 * (x_te_raw - x_min) / (x_max - x_min) - 1.0
    y_te_n = 2.0 * (y_te_raw - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_64 = torch.tensor(x_tr_n[:, None], device=device, dtype=torch.float64)
    D_tr_64 = torch.tensor(y_tr_n[:, None], device=device, dtype=torch.float64)
    X_te_64 = torch.tensor(x_te_n[:, None], device=device, dtype=torch.float64)
    D_te_64 = torch.tensor(y_te_n[:, None], device=device, dtype=torch.float64)

    return {
        "X_train_64": X_tr_64,
        "XT_train_64": X_tr_64.t().contiguous(),
        "D_train_64": D_tr_64,
        "X_test_64": X_te_64,
        "D_test_64": D_te_64,
        "x_tr_raw": x_tr_raw,
        "y_tr_raw": y_tr_raw,
        "x_te_raw": x_te_raw,
        "y_te_raw": y_te_raw,
        "norm_stats": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
    }


def load_or_create_levy_5d_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """ex001 と共通の 5次元 Levy 関数データセットを読み込む．"""
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_levy_5d.npz")
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_all, y_all = d["X_all"], d["y_all"]
    else:
        rng = np.random.default_rng(seed)
        X_all = rng.uniform(-4.0, 4.0, size=(7500, 5)).astype(np.float64)
        t1 = np.sum(((X_all[:, :-1] - 1.0) ** 2) * (1.0 + 10.0 * (np.sin(np.pi * X_all[:, 1:]) ** 2)), axis=1)
        t2 = 10.0 * (np.sin(np.pi * X_all[:, 0]) ** 2)
        t3 = (X_all[:, -1] - 1.0) ** 2
        y_all = (np.pi / 5.0) * (t1 + t2 + t3)
        np.savez(npz_path, X_all=X_all, y_all=y_all)

    X_min, X_max = -4.0, 4.0
    y_min, y_max = float(y_all.min()), float(y_all.max())
    X_norm = 2.0 * (X_all - X_min) / (X_max - X_min) - 1.0
    y_norm = 2.0 * (y_all - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_64 = torch.tensor(X_norm[:5000], device=device, dtype=torch.float64)
    D_tr_64 = torch.tensor(y_norm[:5000, None], device=device, dtype=torch.float64)
    X_te_64 = torch.tensor(X_norm[5000:], device=device, dtype=torch.float64)
    D_te_64 = torch.tensor(y_norm[5000:, None], device=device, dtype=torch.float64)

    return {
        "X_train_64": X_tr_64,
        "XT_train_64": X_tr_64.t().contiguous(),
        "D_train_64": D_tr_64,
        "X_test_64": X_te_64,
        "D_test_64": D_te_64,
    }


def generate_initial_weights(
    num_runs: int, num_params: int, seed: int = 2023, device: Optional[torch.device] = None
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.rand((num_runs, num_params), generator=gen, dtype=torch.float64) - 0.5).to(device)


def staged_parallel_armijo(
    model: BatchedMLP,
    w_base: torch.Tensor,
    g_k: torch.Tensor,
    loss_base: torch.Tensor,
    slope: torch.Tensor,
    X: torch.Tensor,
    D_col: torch.Tensor,
    chi: float = 1e-3,
    tau: float = 0.5,
    ls_max: int = 10,
) -> torch.Tensor:
    Ra, Dw = w_base.shape
    alpha = torch.ones(Ra, device=w_base.device, dtype=w_base.dtype)
    l0 = model.loss_only(w_base + g_k, X, D_col)
    ok0 = l0 <= loss_base + chi * slope
    if ok0.all() or ls_max <= 1:
        return alpha

    rem = torch.nonzero(~ok0).squeeze(-1)
    chunk_size = max(10, ls_max - 1) if ls_max <= 16 else 10
    for start in range(1, ls_max, chunk_size):
        end = min(start + chunk_size, ls_max)
        K = end - start
        cands = tau ** torch.arange(start, end, device=w_base.device, dtype=w_base.dtype)
        wc = w_base[rem].unsqueeze(1) + cands[None, :, None] * g_k[rem].unsqueeze(1)
        lc = model.loss_only(wc.reshape(rem.numel() * K, Dw), X, D_col).view(rem.numel(), K)
        ok_c = lc <= loss_base[rem].unsqueeze(1) + chi * cands.unsqueeze(0) * slope[rem].unsqueeze(1)
        any_c = ok_c.any(dim=1)
        if any_c.any():
            hit = rem[any_c]
            idx_first = torch.argmax(ok_c[any_c].to(torch.int64), dim=1)
            alpha[hit] = cands[idx_first]
        rem = rem[~any_c]
        if rem.numel() == 0:
            break
    if rem.numel() > 0:
        alpha[rem] = tau ** (ls_max - 1)
    return alpha


def _batched_two_loop_direction(
    grad_base: torch.Tensor,
    p_hist: List[torch.Tensor],
    q_hist: List[torch.Tensor],
    pq_hist: List[torch.Tensor],
    valid_hist: List[torch.Tensor],
) -> torch.Tensor:
    """
    記憶制限手法 (LQN, LNAQ, LMoQ) の Two-Loop Recursion を R 試行同時にバッチ並列実行する．
    p_hist, q_hist の各要素は (Ra, Dw)，pq_hist と valid_hist は (Ra,)．
    """
    g = -grad_base.clone()
    n_hist = len(p_hist)
    if n_hist == 0:
        return g

    alphas: List[torch.Tensor] = [torch.zeros(grad_base.shape[0], device=grad_base.device, dtype=grad_base.dtype)] * n_hist
    # Loop 1 (最新 -> 最古)
    for i in range(n_hist - 1, -1, -1):
        v_mask = valid_hist[i]
        pq_s = torch.where(v_mask, pq_hist[i], torch.ones_like(pq_hist[i]))
        phi_i = torch.sum(p_hist[i] * g, dim=1) / pq_s
        phi_i = torch.where(v_mask, phi_i, torch.zeros_like(phi_i))
        alphas[i] = phi_i
        g = g - phi_i[:, None] * q_hist[i]

    # 中央対角スケーリング (最新有効履歴を使用)
    v_last = valid_hist[-1]
    qq_last = torch.sum(q_hist[-1] * q_hist[-1], dim=1)
    safe_qq = torch.where(v_last & (qq_last > 1e-30), qq_last, torch.ones_like(qq_last))
    gamma_k = torch.where(v_last & (qq_last > 1e-30), pq_hist[-1] / safe_qq, torch.ones_like(qq_last))
    g = g * gamma_k[:, None]

    # Loop 2 (最古 -> 最新)
    for i in range(n_hist):
        v_mask = valid_hist[i]
        pq_s = torch.where(v_mask, pq_hist[i], torch.ones_like(pq_hist[i]))
        tau = torch.sum(q_hist[i] * g, dim=1) / pq_s
        tau = torch.where(v_mask, tau, torch.zeros_like(tau))
        g = g + (alphas[i] - tau)[:, None] * p_hist[i]

    return g


def run_limited_or_memoryless_batched(
    algo: str,
    w_init: torch.Tensor,
    X_train: torch.Tensor,
    XT_train: torch.Tensor,
    D_train: torch.Tensor,
    X_test: torch.Tensor,
    D_test: torch.Tensor,
    n_in: int,
    n_hid: int,
    memory_size: int = 8,
    use_adaptive_mu: bool = True,
    mu: float = 0.85,
    gamma: float = 1e-5,
    chi: float = 1e-3,
    tau: float = 0.5,
    ls_max: int = 10,
    max_step_norm: float = 20.0,
    curvature_eps: float = 1e-14,
    use_global_conv: bool = False,
    global_conv_scale: float = 1.0,
    max_mu: float = 0.92,
    eps: float = 1e-6,
    loss_stop: Optional[float] = None,
    max_epochs: int = 50000,
    check_interval: int = 100,
    record_stride: int = 250,
    show_pbar: bool = True,
    desc: Optional[str] = None,
) -> Dict[str, object]:
    """
    記憶制限手法 (LQN, LNAQ, LMoQ) および メモリレス手法 (CG, MLQN, MLNAQ, MLMoQ) を
    R 試行一括 GPU 並列で実行するエンジン．
    """
    valid_all = ("LQN", "LNAQ", "LMoQ", "CG", "MLQN", "MLNAQ", "MLMoQ")
    assert algo in valid_all, f"Unsupported algo {algo}"
    device = X_train.device
    dtype = torch.float64
    model = BatchedMLP(n_in=n_in, n_hid=n_hid, n_out=1)
    R, Dw = w_init.shape

    w = w_init.to(dtype).clone()
    v = torch.zeros_like(w)
    theta_ada = torch.ones(R, device=device, dtype=dtype)

    w_final = w.clone()
    loss_final = torch.zeros(R, device=device, dtype=dtype)
    epochs_final = torch.zeros(R, device=device, dtype=torch.long)
    diverged = torch.zeros(R, device=device, dtype=torch.bool)
    orig_idx = torch.arange(R, device=device)

    is_limited = algo in ("LQN", "LNAQ", "LMoQ")
    is_ml = algo in ("MLQN", "MLNAQ", "MLMoQ")
    is_cg = algo == "CG"
    is_moq = algo in ("LMoQ", "MLMoQ")
    is_naq = algo in ("LNAQ", "MLNAQ")

    p_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    q_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    pq_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    valid_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)

    p_prev = torch.zeros_like(w)
    q_prev = torch.zeros_like(w)
    pq_prev = torch.zeros(R, device=device, dtype=dtype)
    qq_prev = torch.zeros(R, device=device, dtype=dtype)
    has_prev = torch.zeros(R, device=device, dtype=torch.bool)

    g_cg_prev = torch.zeros_like(w)
    y_cg_prev = torch.zeros_like(w)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_train, XT_train, D_train)
    grad_prev = grad_k.clone()

    live_losses = loss_k.clone()
    rec_epochs = [0]
    rec_times = [0.0]
    rec_means = [float(live_losses.mean().item())]
    rec_bests = [float(live_losses.min().item())]

    label = desc or (f"{algo}(m={memory_size})" if is_limited else algo)
    pbar = tqdm(total=max_epochs, desc=label, disable=not show_pbar, mininterval=1.5)
    last_k = 0
    total_step_runs = 0

    for k in range(1, max_epochs + 1):
        if k % check_interval == 1:
            gnorm = torch.norm(grad_k, dim=1)
            bad_num = (~torch.isfinite(loss_k)) | (~torch.isfinite(gnorm)) | (gnorm > 1e3)
            cond_loss = (loss_k <= loss_stop) if (loss_stop is not None) else torch.zeros_like(bad_num)
            conv = (gnorm <= eps) | cond_loss | bad_num
            if conv.any():
                d_idx = orig_idx[conv]
                w_final[d_idx] = w[conv]
                loss_final[d_idx] = loss_k[conv]
                live_losses[d_idx] = loss_k[conv]
                epochs_final[d_idx] = k - 1
                diverged[d_idx] = bad_num[conv]
                alive = ~conv
                if not alive.any():
                    pbar.update(k - 1 - last_k)
                    last_k = k - 1
                    break
                orig_idx = orig_idx[alive]
                w = w[alive]
                v = v[alive]
                theta_ada = theta_ada[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]
                grad_prev = grad_prev[alive]
                if is_limited:
                    p_hist = deque([t[alive] for t in p_hist], maxlen=memory_size)
                    q_hist = deque([t[alive] for t in q_hist], maxlen=memory_size)
                    pq_hist = deque([t[alive] for t in pq_hist], maxlen=memory_size)
                    valid_hist = deque([t[alive] for t in valid_hist], maxlen=memory_size)
                else:
                    p_prev = p_prev[alive]
                    q_prev = q_prev[alive]
                    pq_prev = pq_prev[alive]
                    qq_prev = qq_prev[alive]
                    has_prev = has_prev[alive]
                    g_cg_prev = g_cg_prev[alive]
                    y_cg_prev = y_cg_prev[alive]

        Ra = w.shape[0]
        total_step_runs += Ra
        gnorm_now = torch.norm(grad_k, dim=1)
        cond_loss_now = (loss_k <= loss_stop) if (loss_stop is not None) else torch.zeros(Ra, device=device, dtype=torch.bool)
        still = (gnorm_now > eps) & (~cond_loss_now)

        # 1. mu_k
        if algo in ("LQN", "CG", "MLQN"):
            mu_vec = torch.zeros(Ra, device=device, dtype=dtype)
        elif use_adaptive_mu:
            mu_vec, theta_ada = adaptive_momentum_step_tensor(theta_ada, gamma=gamma)
            mu_vec = torch.clamp(mu_vec, max=max_mu)
        else:
            mu_vec = torch.full((Ra,), mu, device=device, dtype=dtype)

        # 2. 基準点と勾配（外挿勾配）
        if algo in ("LQN", "CG", "MLQN"):
            w_base, loss_base, grad_base = w, loss_k, grad_k
        elif is_naq:
            if float(mu_vec.max().item()) == 0.0:
                w_base, loss_base, grad_base = w, loss_k, grad_k
            else:
                w_base = w + mu_vec[:, None] * v
                loss_base, grad_base = model.loss_and_grad(w_base, X_train, XT_train, D_train)
                over = loss_base > loss_k
                if over.any():
                    v = torch.where(over[:, None], torch.zeros_like(v), v)
                    mu_vec = torch.where(over, torch.zeros_like(mu_vec), mu_vec)
                    if use_adaptive_mu:
                        theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                    w_base = torch.where(over[:, None], w, w_base)
                    loss_base = torch.where(over, loss_k, loss_base)
                    grad_base = torch.where(over[:, None], grad_k, grad_base)
        else:  # LMoQ / MLMoQ
            if k == 1 or float(mu_vec.max().item()) == 0.0:
                w_base, loss_base, grad_base = w, loss_k, grad_k
            else:
                w_base = w + mu_vec[:, None] * v
                loss_base = model.loss_only(w_base, X_train, D_train)
                grad_base = (1.0 + mu_vec[:, None]) * grad_k - mu_vec[:, None] * grad_prev
                over = loss_base > loss_k
                if over.any():
                    v = torch.where(over[:, None], torch.zeros_like(v), v)
                    mu_vec = torch.where(over, torch.zeros_like(mu_vec), mu_vec)
                    if use_adaptive_mu:
                        theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                    w_base = torch.where(over[:, None], w, w_base)
                    loss_base = torch.where(over, loss_k, loss_base)
                    grad_base = torch.where(over[:, None], grad_k, grad_base)

        # 3. 探索方向 g_k
        if is_limited:
            g_raw = _batched_two_loop_direction(
                grad_base, list(p_hist), list(q_hist), list(pq_hist), list(valid_hist)
            )
        elif is_cg:
            if k == 1:
                g_raw = -grad_base
            else:
                gy = torch.sum(grad_base * y_cg_prev, dim=1)
                denom = torch.sum(g_cg_prev * y_cg_prev, dim=1)
                safe_d = torch.where(torch.abs(denom) > 1e-20, denom, torch.ones_like(denom))
                beta_hs = torch.where(has_prev & (torch.abs(denom) > 1e-20), gy / safe_d, torch.zeros_like(gy))
                beta_hs = torch.clamp(beta_hs, min=0.0)
                g_raw = -grad_base + beta_hs[:, None] * g_cg_prev
        else:  # MLQN / MLNAQ / MLMoQ
            use_prev = has_prev & (pq_prev > curvature_eps) & (qq_prev > 1e-30)
            pq_s = torch.where(use_prev, pq_prev, torch.ones_like(pq_prev))
            qq_s = torch.where(use_prev, qq_prev, torch.ones_like(qq_prev))
            beta_k = torch.sum(p_prev * grad_base, dim=1) / pq_s
            psi_k = torch.sum(q_prev * grad_base, dim=1) / pq_s
            omega_k = qq_s / pq_s
            tau_k = pq_s / qq_s
            g_ml = (
                -tau_k[:, None] * (grad_base - (beta_k[:, None] * q_prev + psi_k[:, None] * p_prev))
                - ((1.0 + tau_k * omega_k) * beta_k)[:, None] * p_prev
            )
            g_raw = torch.where(use_prev[:, None], g_ml, -grad_base)

        g_k = g_raw / torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
        slope = torch.sum(grad_base * g_k, dim=1)
        bad = slope >= 0.0
        if bad.any():
            g_fb = -grad_base / torch.clamp(torch.norm(grad_base, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = torch.where(bad[:, None], g_fb, g_k)
            slope = torch.sum(grad_base * g_k, dim=1)
            if is_ml or is_cg:
                has_prev = has_prev & (~bad)

        # 4. Armijo 直線探索 (論文 Algorithm 10 準拠，最小ステップ tau**(ls_max-1))
        alpha = staged_parallel_armijo(
            model, w_base, g_k, loss_base, slope, X_train, D_train, chi=chi, tau=tau, ls_max=ls_max
        )

        # 5. 更新
        v_next = mu_vec[:, None] * v + alpha[:, None] * g_k
        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_train, XT_train, D_train)

        # 6. 履歴更新
        if is_cg:
            g_cg_prev = torch.where(still[:, None], g_k, g_cg_prev)
            y_cg_prev = torch.where(still[:, None], grad_next - grad_k, y_cg_prev)
            has_prev = has_prev | still
        else:
            p_k = alpha[:, None] * g_k
            delta = grad_next - grad_base
            if use_global_conv:
                p_norm_sq = torch.sum(p_k * p_k, dim=1)
                g_norm_sq = torch.sum(grad_base * grad_base, dim=1)
                g_norm = torch.sqrt(g_norm_sq)
                omega = torch.where(g_norm_sq > 1e-2, torch.full_like(g_norm_sq, 2.0), torch.full_like(g_norm_sq, 100.0)) * global_conv_scale
                dp = torch.sum(delta * p_k, dim=1)
                p_ns_safe = torch.where(p_norm_sq > 1e-30, p_norm_sq, torch.ones_like(p_norm_sq))
                xi_k = omega * g_norm + torch.clamp(-dp / p_ns_safe, min=0.0)
                q_k = torch.where((p_norm_sq > 1e-30)[:, None], delta + xi_k[:, None] * p_k, delta)
            else:
                q_k = delta
            pq = torch.sum(p_k * q_k, dim=1)
            qq = torch.sum(q_k * q_k, dim=1)
            ok_pair = still & (pq > curvature_eps) & (qq > 1e-30)
            if is_limited:
                p_hist.append(p_k)
                q_hist.append(q_k)
                pq_hist.append(pq)
                valid_hist.append(ok_pair)
            else:
                p_prev = torch.where(ok_pair[:, None], p_k, p_prev)
                q_prev = torch.where(ok_pair[:, None], q_k, q_prev)
                pq_prev = torch.where(ok_pair, pq, pq_prev)
                qq_prev = torch.where(ok_pair, qq, qq_prev)
                has_prev = torch.where(still, ok_pair, has_prev)

        grad_prev = torch.where(still[:, None], grad_k, grad_prev)
        w = torch.where(still[:, None], w_next, w)
        v = torch.where(still[:, None], v_next, v)
        loss_k = torch.where(still, loss_next, loss_k)
        grad_k = torch.where(still[:, None], grad_next, grad_k)

        if k % record_stride == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            live_losses[orig_idx] = loss_k
            rec_epochs.append(k)
            rec_times.append(time.perf_counter() - t0)
            rec_means.append(float(live_losses.mean().item()))
            rec_bests.append(float(live_losses.min().item()))

        if show_pbar and (k - last_k >= 1000):
            pbar.update(k - last_k)
            pbar.set_postfix(active=int(w.shape[0]), E_ave=f"{live_losses.mean().item()*1e3:.4f}")
            last_k = k

    if orig_idx.numel() > 0:
        gnorm = torch.norm(grad_k, dim=1)
        bad_num = (~torch.isfinite(loss_k)) | (~torch.isfinite(gnorm)) | (gnorm > 1e3)
        cond_loss = (loss_k <= loss_stop) if (loss_stop is not None) else torch.zeros_like(bad_num)
        still = (gnorm > eps) & (~cond_loss) & (~bad_num)
        ep_rem = torch.where(still, torch.full_like(orig_idx, max_epochs), torch.full_like(orig_idx, k))
        w_final[orig_idx] = w
        loss_final[orig_idx] = loss_k
        epochs_final[orig_idx] = ep_rem
        diverged[orig_idx] = bad_num

    if show_pbar and last_k < max_epochs:
        pbar.update(max_epochs - last_k)
    pbar.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    with torch.no_grad():
        loss_test = model.loss_only(w_final, X_test, D_test)

    e_tr = (loss_final * 1000.0).cpu().numpy()
    e_te = (loss_test * 1000.0).cpu().numpy()
    ep_np = epochs_final.cpu().numpy()
    div_np = diverged.cpu().numpy()
    if loss_stop is not None:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (loss_final.cpu().numpy() <= loss_stop))
    else:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (e_tr < 1.5))
    conv_rate = 100.0 * float(np.mean(conv_mask))
    ave_ep = float(ep_np.mean())
    per_iter_ms = (total_time / max(total_step_runs, 1)) * 1000.0
    est_time = (per_iter_ms / 1000.0) * ave_ep

    return {
        "algo": algo,
        "m": str(memory_size) if is_limited else "-",
        "train_median": float(np.median(e_tr)),
        "train_ave": float(np.mean(e_tr)),
        "train_best": float(np.min(e_tr)),
        "train_worst": float(np.max(e_tr)),
        "test_median": float(np.median(e_te)),
        "test_ave": float(np.mean(e_te)),
        "test_best": float(np.min(e_te)),
        "test_worst": float(np.max(e_te)),
        "ave_epochs": ave_ep,
        "ave_time_sec": est_time,
        "per_iter_ms": per_iter_ms,
        "batch_time_sec": total_time,
        "conv_rate": conv_rate,
        "rec_epochs": np.array(rec_epochs, dtype=np.int64),
        "rec_times": np.array(rec_times, dtype=np.float64),
        "rec_means": np.array(rec_means, dtype=np.float64),
        "rec_bests": np.array(rec_bests, dtype=np.float64),
        "best_weights": w_final[int(np.argmin(e_tr))].detach().clone(),
    }


def format_ex2_dataframe(results: List[Dict[str, object]], precision: int = 3) -> pd.DataFrame:
    fmt = f"{{:.{precision}f}}"
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r["algo"],
            "m": r["m"],
            "E_train (x10^3) Med/Ave/Best/Worst": " / ".join([
                fmt.format(r["train_median"]),
                fmt.format(r["train_ave"]),
                fmt.format(r["train_best"]),
                fmt.format(r["train_worst"]),
            ]),
            "Est. Time(s)": round(r["ave_time_sec"], 3),
            "Per Iter(ms)": round(r["per_iter_ms"], 4),
            "Ave. Iters": int(round(r["ave_epochs"])),
            "E_test (x10^3) Med/Ave/Best/Worst": " / ".join([
                fmt.format(r["test_median"]),
                fmt.format(r["test_ave"]),
                fmt.format(r["test_best"]),
                fmt.format(r["test_worst"]),
            ]),
            "Conv(%)": round(r["conv_rate"], 1),
        })
    return pd.DataFrame(rows)
