"""
ex001/utils.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験1 (第2〜4章) に使用するアルゴリズム・ベンチマーク関数群．

実装内容:
1. 適応的慣性係数 mu_k の更新 (式(64), (65))
2. PyTorch 標準 torch.optim.Optimizer 継承クラス MoQFamilyOptimizer
   (QN, MSQN, NAQ, AdaNAQ, MoQ, AdaMoQ を単一クラスでサポート)
3. Beale 関数 (式(24)) における1次・2次手法の最適化軌跡計算
4. 非線形関数近似問題1 f1(x, a) (式(68), |Tr|=3320, |Te|=6600) および
   非線形関数近似問題2 5次元 Levy 関数 (式(102), |Tr|=5000) のデータセット生成・読み込み
5. 10独立試行一括 GPU 並列学習エンジン (1次手法群・2次手法群)
"""

import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

# Matplotlib 一時キャッシュ警告回避のための設定
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
    """Matplotlib で日本語フォントとマイナス記号が正しく描画されるように設定する．"""
    plt.rcParams["font.family"] = ["Meiryo", "Yu Gothic", "MS Gothic", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def adaptive_momentum_step_scalar(theta: float, gamma: float = 1e-5) -> Tuple[float, float]:
    """
    論文 式(64), (65) に基づき適応的慣性係数 mu_k と次ステップの theta_{k+1} を求める．
    theta_{k+1}^2 = (1 - theta_{k+1}) * theta_k^2 + gamma * theta_{k+1}
    => theta_{k+1}^2 + (theta_k^2 - gamma) * theta_{k+1} - theta_k^2 = 0 の正根を解析的に算出．
    """
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + math.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def adaptive_momentum_step_tensor(
    theta: torch.Tensor, gamma: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """バッチテンソル版: 式(64), (65) により各試行の mu_k と theta_{k+1} を一括計算する．"""
    t2 = theta * theta
    b = t2 - gamma
    theta_next = 0.5 * (-b + torch.sqrt(b * b + 4.0 * t2))
    mu_k = (theta * (1.0 - theta)) / (t2 + theta_next)
    return mu_k, theta_next


def _compute_global_conv_q_scalar(
    delta: torch.Tensor,
    p_k: torch.Tensor,
    g_lin: torch.Tensor,
    use_global_conv: bool = True,
    omega_scale: float = 1.0,
) -> torch.Tensor:
    """
    論文 式(98)-(100) に基づくグローバル収束項 q_hat_k = delta_k + xi_k * p_k の計算．
    Li & Fukushima (2001) に基づき，曲率条件 p_k^T q_hat_k > 0 を常に保証する．
    """
    if not use_global_conv:
        return delta
    p_norm_sq = float(torch.dot(p_k, p_k).item())
    if p_norm_sq <= 1e-30:
        return delta
    g_norm_sq = float(torch.dot(g_lin, g_lin).item())
    g_norm = math.sqrt(g_norm_sq)
    omega = 2.0 if g_norm_sq > 1e-2 else 100.0
    omega *= omega_scale
    dp = float(torch.dot(delta, p_k).item())
    xi_k = omega * g_norm + max(-dp / p_norm_sq, 0.0)
    return delta + xi_k * p_k


class MoQFamilyOptimizer(Optimizer):
    """
    論文 第2〜4章の準ニュートン系最適化手法を統一的に実装した汎用 PyTorch Optimizer．
    サポートするアルゴリズム:
      - 'QN'     : 準ニュートン法 (BFGS, Algorithm 8)
      - 'MSQN'   : マルチステップ準ニュートン法 (m=2, Algorithm 11)
      - 'NAQ'    : ネステロフの加速準ニュートン法 (固定 mu, Algorithm 9)
      - 'AdaNAQ' : 適応的 NAQ (式(64),(65) による適応的 mu_k)
      - 'MoQ'    : 慣性付準ニュートン法 (固定 mu, 勾配外挿近似, Algorithm 12)
      - 'AdaMoQ' : 適応的 MoQ (式(64),(65) による適応的 mu_k + 勾配外挿近似)

    MoQ / AdaMoQ では先読み点 w_k + mu_k * v_k における勾配計算を省き，
    (1 + mu_k) * grad E(w_k) - mu_k * grad E(w_{k-1}) の線形外挿を用いるため，
    1反復あたりの誤差逆伝播は更新後の点 w_{k+1} における 1 回のみとなる．
    """

    VALID_ALGOS = ("QN", "MSQN", "NAQ", "AdaNAQ", "MoQ", "AdaMoQ")

    def __init__(
        self,
        params,
        algo: str = "AdaMoQ",
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
            raise ValueError(f"Unsupported algo '{algo}'. Must be one of {self.VALID_ALGOS}")
        if not 0.0 <= mu < 1.0:
            raise ValueError(f"Invalid momentum mu={mu}")
        defaults = dict(
            algo=algo,
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
        self._params_list: List[torch.nn.Parameter] = []
        for group in self.param_groups:
            self._params_list.extend(group["params"])

    def _gather_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self._params_list], dim=0)

    def _set_flat_params(self, flat_w: torch.Tensor) -> None:
        offset = 0
        for p in self._params_list:
            numel = p.numel()
            p.data.copy_(flat_w[offset : offset + numel].view_as(p.data))
            offset += numel

    def _gather_flat_grad(self) -> torch.Tensor:
        grads = []
        for p in self._params_list:
            if p.grad is None:
                grads.append(torch.zeros_like(p.data).view(-1))
            else:
                grads.append(p.grad.data.view(-1))
        return torch.cat(grads, dim=0)

    @torch.no_grad()
    def step(self, closure: Callable[[bool], torch.Tensor]) -> Tuple[float, float, float, float]:
        """
        1反復を実行する．
        closure(backward=True) : 順伝播＋逆伝播を行い損失テンソルを返す．
        closure(backward=False): 順伝播のみを行い損失テンソルを返す（Armijo探索・MoQ先読み評価用）．

        Returns
        -------
        loss_next, grad_norm_next, alpha, mu_k
        """
        cfg = self.param_groups[0]
        algo: str = cfg["algo"]
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
        dim_w = w_k.numel()
        state = self.state["global_moq_state"]

        # 初回ステップの初期化 (k = 1)
        if len(state) == 0:
            state["step"] = 0
            state["theta"] = 1.0
            state["v"] = torch.zeros_like(w_k)
            state["H"] = torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype)
            with torch.enable_grad():
                loss_init = closure(True)
            g_init = self._gather_flat_grad().clone()
            state["loss_k"] = float(loss_init.item())
            state["grad_k"] = g_init
            state["grad_prev"] = g_init.clone()
            state["w_prev"] = w_k.clone()
            state["s_hat_1"] = torch.zeros_like(w_k)
            state["s_hat_2"] = torch.zeros_like(w_k)

        state["step"] += 1
        k: int = state["step"]
        v_k: torch.Tensor = state["v"]
        H_k: torch.Tensor = state["H"]
        loss_k: float = state["loss_k"]
        grad_k: torch.Tensor = state["grad_k"]
        grad_prev: torch.Tensor = state["grad_prev"]

        # 1. 慣性係数 mu_k の決定
        if algo in ("QN", "MSQN"):
            mu_k = 0.0
        elif algo in ("AdaNAQ", "AdaMoQ"):
            mu_k, theta_next = adaptive_momentum_step_scalar(state["theta"], gamma=gamma)
            mu_k = min(mu_k, max_mu)
            state["theta"] = theta_next
        else:
            mu_k = mu_fixed

        # 2. 探索基準点と勾配（または外挿勾配）の計算
        if algo in ("QN", "MSQN"):
            w_base = w_k
            loss_base = loss_k
            grad_base = grad_k
        elif algo in ("NAQ", "AdaNAQ"):
            if mu_k == 0.0:
                w_base = w_k
                loss_base = loss_k
                grad_base = grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                with torch.enable_grad():
                    loss_base_t = closure(True)
                loss_base = float(loss_base_t.item())
                grad_base = self._gather_flat_grad().clone()
                # Adaptive Restart (O'Donoghue & Candes): 先読み点で損失増大時に慣性をリセット
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if algo == "AdaNAQ":
                        state["theta"] = 1.0
                    w_base = w_k
                    loss_base = loss_k
                    grad_base = grad_k
        else:  # MoQ / AdaMoQ: 勾配計算なし！損失のみ順伝播で評価し外挿勾配を使用
            if mu_k == 0.0 or k == 1:
                w_base = w_k
                loss_base = loss_k
                grad_base = grad_k
            else:
                w_base = w_k + mu_k * v_k
                self._set_flat_params(w_base)
                loss_base = float(closure(False).item())
                grad_base = (1.0 + mu_k) * grad_k - mu_k * grad_prev
                if loss_base > loss_k:
                    v_k.zero_()
                    mu_k = 0.0
                    if algo == "AdaMoQ":
                        state["theta"] = 1.0
                    w_base = w_k
                    loss_base = loss_k
                    grad_base = grad_k

        # 3. 探索方向 g_k = -H_k * grad_base の計算と安全スケーリング
        g_raw = -torch.mv(H_k, grad_base)
        gn = float(torch.norm(g_raw).item())
        scale = max(1.0, gn / max_step_norm)
        g_k = g_raw / scale

        slope = float(torch.dot(grad_base, g_k).item())
        if slope >= 0.0:
            H_k.copy_(torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype))
            g_raw = -grad_base
            scale = max(1.0, float(torch.norm(g_raw).item()) / max_step_norm)
            g_k = g_raw / scale
            slope = float(torch.dot(grad_base, g_k).item())

        # 4. Armijo 条件によるステップサイズ探索（論文 Algorithm 10 準拠，最小ステップ tau**(ls_max-1)）
        alpha = tau ** (ls_max - 1)
        for m in range(ls_max):
            cand_alpha = tau ** m
            self._set_flat_params(w_base + cand_alpha * g_k)
            loss_try = float(closure(False).item())
            if loss_try <= loss_base + chi * cand_alpha * slope:
                alpha = cand_alpha
                break

        # 5. 重み更新と更新後点 w_{k+1} の勾配計算（MoQ ではこの 1 回のみ）
        v_next = mu_k * v_k + alpha * g_k
        w_next = w_k + v_next
        self._set_flat_params(w_next)

        with torch.enable_grad():
            loss_next_t = closure(True)
        loss_next = float(loss_next_t.item())
        grad_next = self._gather_flat_grad().clone()

        # 6. セカントベクトルと逆ヘッセ近似行列 H_{k+1} の BFGS 更新
        if algo == "MSQN":
            if k <= 2:
                nu_k = 0.0
            else:
                n1 = float(torch.norm(state["s_hat_1"]).item())
                n2 = float(torch.norm(state["s_hat_2"]).item())
                denom = 2.0 * (n2 * n2) + n1 * n2
                nu_k = (n1 * n1) / denom if denom > 1e-30 else 0.0
            w_prev = state["w_prev"]
            p_k = w_next - (1.0 + nu_k) * w_k + nu_k * w_prev
            delta = grad_next - (1.0 + nu_k) * grad_k + nu_k * grad_prev
            g_lin = grad_k
            state["s_hat_2"] = state["s_hat_1"]
            state["s_hat_1"] = p_k.clone()
        else:
            p_k = alpha * g_k  # = w_{k+1} - (w_k + mu_k * v_k)
            delta = grad_next - grad_base
            g_lin = grad_base

        q_k = _compute_global_conv_q_scalar(delta, p_k, g_lin, use_global_conv=use_gc, omega_scale=gc_scale)
        pq = float(torch.dot(p_k, q_k).item())
        if pq > curv_eps:
            Hq = torch.mv(H_k, q_k)
            qHq = float(torch.dot(q_k, Hq).item())
            term1 = (torch.outer(Hq, p_k) + torch.outer(p_k, Hq)) / pq
            term2 = (1.0 + qHq / pq) * torch.outer(p_k, p_k) / pq
            H_k.add_(-term1 + term2)
        else:
            H_k.copy_(torch.eye(dim_w, device=w_k.device, dtype=w_k.dtype))

        state["w_prev"] = w_k
        state["grad_prev"] = grad_k
        state["v"] = v_next
        state["loss_k"] = loss_next
        state["grad_k"] = grad_next

        return loss_next, float(torch.norm(grad_next).item()), alpha, mu_k


# =====================================================================
# Beale 関数ベンチマーク (論文 2.3節 式(24), 図8〜17)
# =====================================================================

def beale_loss_and_grad(xy: np.ndarray) -> Tuple[float, np.ndarray]:
    """論文 式(24): Beale 関数 f(x, y) と解析的勾配．大域的最小解は (3.0, 0.5)．"""
    x, y = float(xy[0]), float(xy[1])
    t1 = 1.5 - x + x * y
    t2 = 2.25 - x + x * (y ** 2)
    t3 = 2.625 - x + x * (y ** 3)
    val = t1 * t1 + t2 * t2 + t3 * t3
    gx = 2.0 * t1 * (y - 1.0) + 2.0 * t2 * (y * y - 1.0) + 2.0 * t3 * (y ** 3 - 1.0)
    gy = 2.0 * t1 * x + 2.0 * t2 * (2.0 * x * y) + 2.0 * t3 * (3.0 * x * y * y)
    return val, np.array([gx, gy], dtype=np.float64)


def run_beale_trajectories(
    x0: float = 7.0,
    y0: float = 1.4,
    eps: float = 1e-6,
    k_max: int = 1000,
) -> Dict[str, Dict[str, object]]:
    """
    論文 2.3節 (x0=7.0, y0=1.4, ただし図表示範囲に合わせ軌道を取得) における
    1次手法 (GD, CM, NAG, AdaGrad, RMSprop, AdaDelta, Adam) と
    2次手法 (QN, NAQ, MoQ, AdaMoQ) の最適化軌跡を計算する．
    """
    results: Dict[str, Dict[str, object]] = {}

    def _run_1st(name: str, lr: float, mu: float = 0.0, beta1: float = 0.9, beta2: float = 0.999) -> None:
        w = np.array([x0, y0], dtype=np.float64)
        v = np.zeros(2, dtype=np.float64)
        G = np.zeros(2, dtype=np.float64)
        r = np.zeros(2, dtype=np.float64)
        theta = np.zeros(2, dtype=np.float64)
        lam = 1e-8
        traj = [w.copy()]
        for k in range(1, k_max + 1):
            val, grad = beale_loss_and_grad(w)
            if np.linalg.norm(grad) <= eps:
                break
            if name == "GD":
                v = -lr * grad
            elif name == "CM":
                v = mu * v - lr * grad
            elif name == "NAG":
                _, g_nes = beale_loss_and_grad(w + mu * v)
                v = mu * v - lr * g_nes
            elif name == "AdaGrad":
                G += grad ** 2
                v = -lr * grad / np.sqrt(G + lam)
            elif name == "RMSprop":
                theta = beta1 * theta + (1.0 - beta1) * (grad ** 2)
                v = -lr * grad / np.sqrt(theta + lam)
            elif name == "AdaDelta":
                lam_ad = 1e-6
                theta = beta1 * theta + (1.0 - beta1) * (grad ** 2)
                v = -(np.sqrt(r + lam_ad) / np.sqrt(theta + lam_ad)) * grad
                r = beta1 * r + (1.0 - beta1) * (v ** 2)
            elif name == "Adam":
                r = beta1 * r + (1.0 - beta1) * grad
                theta = beta2 * theta + (1.0 - beta2) * (grad ** 2)
                r_hat = r / (1.0 - beta1 ** k)
                th_hat = theta / (1.0 - beta2 ** k)
                v = -lr * r_hat / (np.sqrt(th_hat) + lam)
            # 発散防止用の安全チェック
            w_next = w + v
            if np.linalg.norm(w_next) > 100.0 or not np.all(np.isfinite(w_next)):
                break
            w = w_next
            traj.append(w.copy())
        results[name] = {"traj": np.array(traj), "iters": len(traj) - 1, "final_w": w, "final_f": beale_loss_and_grad(w)[0]}

    # 1次手法の実行 (高急峻域での発散を防ぐ学習率設定)
    _run_1st("GD", lr=0.0002)
    _run_1st("CM", lr=0.0002, mu=0.9)
    _run_1st("NAG", lr=0.0002, mu=0.9)
    _run_1st("AdaGrad", lr=0.2)
    _run_1st("RMSprop", lr=0.2, beta1=0.9)
    _run_1st("AdaDelta", lr=0.0, beta1=0.6)
    _run_1st("Adam", lr=0.1, beta1=0.9, beta2=0.999)

    # 2次手法の実行 (QN, NAQ, MoQ, AdaMoQ)
    for algo in ("QN", "NAQ", "MoQ", "AdaMoQ"):
        w = np.array([x0, y0], dtype=np.float64)
        v = np.zeros(2, dtype=np.float64)
        H = np.eye(2, dtype=np.float64)
        f_k, grad_k = beale_loss_and_grad(w)
        grad_prev = grad_k.copy()
        theta_ada = 1.0
        traj = [w.copy()]
        for k in range(1, k_max + 1):
            if np.linalg.norm(grad_k) <= eps:
                break
            if algo == "QN":
                mu_k = 0.0
                w_base, f_base, g_base = w, f_k, grad_k
            elif algo == "NAQ":
                mu_k = 0.8
                w_base = w + mu_k * v
                f_base, g_base = beale_loss_and_grad(w_base)
            elif algo == "MoQ":
                mu_k = 0.8 if k > 1 else 0.0
                w_base = w + mu_k * v
                f_base, _ = beale_loss_and_grad(w_base)
                g_base = (1.0 + mu_k) * grad_k - mu_k * grad_prev
            else:
                mu_k, theta_ada = adaptive_momentum_step_scalar(theta_ada, gamma=1e-5)
                mu_k = min(mu_k, 0.85)
                if k == 1:
                    w_base, f_base, g_base = w, f_k, grad_k
                else:
                    w_base = w + mu_k * v
                    f_base, _ = beale_loss_and_grad(w_base)
                    g_base = (1.0 + mu_k) * grad_k - mu_k * grad_prev

            if mu_k > 0.0 and f_base > f_k:
                v = np.zeros(2, dtype=np.float64)
                mu_k = 0.0
                theta_ada = 1.0
                w_base, f_base, g_base = w, f_k, grad_k

            g_k = -H @ g_base
            gn = np.linalg.norm(g_k)
            if gn > 20.0:
                g_k = g_k * (20.0 / gn)
            slope = float(np.dot(g_base, g_k))
            if slope >= 0.0:
                H = np.eye(2, dtype=np.float64)
                g_k = -g_base / max(1.0, np.linalg.norm(g_base) / 20.0)
                slope = float(np.dot(g_base, g_k))

            alpha = 0.5 ** 9
            for m in range(10):
                ca = 0.5 ** m
                f_try, _ = beale_loss_and_grad(w_base + ca * g_k)
                if f_try <= f_base + 1e-3 * ca * slope:
                    alpha = ca
                    break

            v_next = mu_k * v + alpha * g_k
            w_next = w + v_next
            f_next, grad_next = beale_loss_and_grad(w_next)
            p_k = alpha * g_k
            q_k = grad_next - g_base
            pq = float(np.dot(p_k, q_k))
            if pq > 1e-14:
                Hq = H @ q_k
                qHq = float(np.dot(q_k, Hq))
                H = H - (np.outer(Hq, p_k) + np.outer(p_k, Hq)) / pq + (1.0 + qHq / pq) * np.outer(p_k, p_k) / pq
            else:
                H = np.eye(2, dtype=np.float64)
            grad_prev = grad_k
            grad_k = grad_next
            f_k = f_next
            v = v_next
            w = w_next
            traj.append(w.copy())
        results[algo] = {"traj": np.array(traj), "iters": len(traj) - 1, "final_w": w, "final_f": beale_loss_and_grad(w)[0]}

    return results


# =====================================================================
# データセット生成・読み込み (f1 2D関数近似, 5次元 Levy 関数)
# =====================================================================

def target_function_f1_2d(x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """論文 式(68): f1(x, a) = 1 + (x + 2x^2) * sin(a * x^2), x in [-4, 4), a in [-1, 1)."""
    return 1.0 + (x + 2.0 * (x ** 2)) * np.sin(a * (x ** 2))


def load_or_create_f1_2d_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    実験1A用データセット:
      訓練データ |Tr| = 3,320 (83 x 40 格子点, x in [-4, 4), a in [-1, 1))
      テストデータ |Te| = 6,600 (一様乱数サンプリング + a=-1評価用断面)
      論文 3.2節・4.3節 に従い入力・出力を [-1.0, 1.0] に線形正規化して返す．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_ex1_f1_2d.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_tr_raw, y_tr_raw = d["X_tr_raw"], d["y_tr_raw"]
        X_te_raw, y_te_raw = d["X_te_raw"], d["y_te_raw"]
    else:
        rng = np.random.default_rng(seed)
        x_grid = np.linspace(-4.0, 4.0, 83, endpoint=False, dtype=np.float64)
        a_grid = np.linspace(-1.0, 1.0, 40, endpoint=False, dtype=np.float64)
        Xg, Ag = np.meshgrid(x_grid, a_grid, indexing="ij")
        X_tr_raw = np.stack([Xg.ravel(), Ag.ravel()], axis=1)  # (3320, 2)
        y_tr_raw = target_function_f1_2d(X_tr_raw[:, 0], X_tr_raw[:, 1])

        x_te = rng.uniform(-4.0, 4.0, size=6600).astype(np.float64)
        a_te = rng.uniform(-1.0, 1.0, size=6600).astype(np.float64)
        X_te_raw = np.stack([x_te, a_te], axis=1)
        y_te_raw = target_function_f1_2d(X_te_raw[:, 0], X_te_raw[:, 1])

        np.savez(npz_path, X_tr_raw=X_tr_raw, y_tr_raw=y_tr_raw, X_te_raw=X_te_raw, y_te_raw=y_te_raw)

    x_min = np.array([-4.0, -1.0], dtype=np.float64)
    x_max = np.array([4.0, 1.0], dtype=np.float64)
    y_min = min(float(y_tr_raw.min()), float(y_te_raw.min()))
    y_max = max(float(y_tr_raw.max()), float(y_te_raw.max()))

    X_tr_n = 2.0 * (X_tr_raw - x_min) / (x_max - x_min) - 1.0
    X_te_n = 2.0 * (X_te_raw - x_min) / (x_max - x_min) - 1.0
    y_tr_n = 2.0 * (y_tr_raw - y_min) / (y_max - y_min) - 1.0
    y_te_n = 2.0 * (y_te_raw - y_min) / (y_max - y_min) - 1.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_64 = torch.tensor(X_tr_n, device=device, dtype=torch.float64)
    D_tr_64 = torch.tensor(y_tr_n[:, None], device=device, dtype=torch.float64)
    X_te_64 = torch.tensor(X_te_n, device=device, dtype=torch.float64)
    D_te_64 = torch.tensor(y_te_n[:, None], device=device, dtype=torch.float64)

    return {
        "X_train_64": X_tr_64,
        "XT_train_64": X_tr_64.t().contiguous(),
        "D_train_64": D_tr_64,
        "X_test_64": X_te_64,
        "D_test_64": D_te_64,
        "X_train_32": X_tr_64.to(torch.float32),
        "XT_train_32": X_tr_64.t().contiguous().to(torch.float32),
        "D_train_32": D_tr_64.to(torch.float32),
        "norm_stats": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
    }


def levy_function_paper_5d(X_raw: np.ndarray) -> np.ndarray:
    """
    論文 式(102) に定義された 5次元 Levy 関数:
    f(x_1...x_d) = (pi / d) * { sum_{l=1}^{d-1} [(x_l - 1)^2 * (1 + 10 sin^2(pi * x_{l+1}))]
                                + 10 sin^2(pi * x_1) + (x_d - 1)^2 },  x_l in [-4, 4]
    """
    d = X_raw.shape[1]
    term1 = np.sum(((X_raw[:, :-1] - 1.0) ** 2) * (1.0 + 10.0 * (np.sin(np.pi * X_raw[:, 1:]) ** 2)), axis=1)
    term2 = 10.0 * (np.sin(np.pi * X_raw[:, 0]) ** 2)
    term3 = (X_raw[:, -1] - 1.0) ** 2
    return (np.pi / float(d)) * (term1 + term2 + term3)


def load_or_create_levy_5d_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    実験1B・実験2B用データセット (論文 4.3.2節・5.4.2節):
      訓練データ |Tr| = 5,000 点，独立テストデータ |Te| = 2,500 点，x_l in [-4, 4] 一様乱数．
      [-1.0, 1.0] に正規化して返す．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_levy_5d.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_all = d["X_all"]
        y_all = d["y_all"]
    else:
        rng = np.random.default_rng(seed)
        X_all = rng.uniform(-4.0, 4.0, size=(7500, 5)).astype(np.float64)
        y_all = levy_function_paper_5d(X_all).astype(np.float64)
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
        "X_train_32": X_tr_64.to(torch.float32),
        "XT_train_32": X_tr_64.t().contiguous().to(torch.float32),
        "D_train_32": D_tr_64.to(torch.float32),
    }


def generate_initial_weights(
    num_runs: int, num_params: int, seed: int = 2023, device: Optional[torch.device] = None
) -> torch.Tensor:
    """R 試行分の初期重み (R, D_w) を [-0.5, 0.5] の一様乱数で生成する．"""
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
    """
    2段階並列 Armijo バックトラッキング直線探索 (論文 Algorithm 10 準拠):
      Stage 1: 全試行で alpha = 1.0 を一括評価
      Stage 2: 棄却された試行のみ残りの候補 (ls_max=10 の場合は単一バッチ) を評価し，
               全棄却時は最小ステップサイズ tau ** (ls_max - 1) を返却する．
    """
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


def run_first_order_batched(
    algo: str,
    w_init: torch.Tensor,
    X_train: torch.Tensor,
    XT_train: torch.Tensor,
    D_train: torch.Tensor,
    X_test: torch.Tensor,
    D_test: torch.Tensor,
    n_in: int,
    n_hid: int,
    mu: float = 0.9,
    lr: float = 0.01,
    chi: float = 1e-3,
    tau: float = 0.5,
    ls_max: int = 10,
    max_step_norm: float = 20.0,
    max_mu: float = 0.92,
    eps: float = 1e-6,
    loss_stop: Optional[float] = None,
    max_epochs: int = 50000,
    record_stride: int = 500,
    show_pbar: bool = True,
    desc: Optional[str] = None,
) -> Dict[str, object]:
    """
    1次手法 (GD, CM, NAG, OptNAG, AdaGrad, RMSprop, AdaDelta, Adam) の R 試行一括 GPU 並列学習．
    論文 3.2節・4.3節 に準拠しバッチ学習を行う．
    """
    device = X_train.device
    dtype = X_train.dtype
    model = BatchedMLP(n_in=n_in, n_hid=n_hid, n_out=1)
    R, Dw = w_init.shape

    w = w_init.to(dtype).clone()
    v = torch.zeros_like(w)
    r_state = torch.zeros_like(w)
    theta_state = torch.zeros_like(w)
    theta_ada = torch.ones(R, device=device, dtype=dtype)

    w_final = w.clone()
    loss_final = torch.zeros(R, device=device, dtype=dtype)
    epochs_final = torch.zeros(R, device=device, dtype=torch.long)
    diverged = torch.zeros(R, device=device, dtype=torch.bool)
    orig_idx = torch.arange(R, device=device)

    use_armijo = algo in ("GD", "CM", "NAG", "OptNAG")
    is_nesterov = algo in ("NAG", "OptNAG")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_train, XT_train, D_train)
    live_losses = loss_k.clone()
    rec_epochs = [0]
    rec_means = [float(live_losses.mean().item())]

    label = desc or f"{algo} (mu={mu:.2f})" if algo in ("CM", "NAG") else (desc or algo)
    pbar = tqdm(total=max_epochs, desc=label, disable=not show_pbar, mininterval=1.5)
    last_k = 0
    check_interval = 250

    for k in range(1, max_epochs + 1):
        if k % check_interval == 1:
            gnorm = torch.norm(grad_k, dim=1)
            bad_num = (~torch.isfinite(loss_k)) | (~torch.isfinite(gnorm)) | (gnorm > 1e3)
            conv = (gnorm <= eps) | bad_num
            if loss_stop is not None:
                conv = conv | ((loss_k <= loss_stop) & torch.isfinite(loss_k))
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
                r_state = r_state[alive]
                theta_state = theta_state[alive]
                theta_ada = theta_ada[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]

        Ra = w.shape[0]
        if algo == "OptNAG":
            mu_vec, theta_ada = adaptive_momentum_step_tensor(theta_ada, gamma=1e-5)
            mu_vec = torch.clamp(mu_vec, max=max_mu)
            mu_col = mu_vec[:, None]
        elif is_nesterov or algo == "CM":
            mu_col = torch.full((Ra, 1), mu, device=device, dtype=dtype)
        else:
            mu_col = torch.zeros((Ra, 1), device=device, dtype=dtype)

        if is_nesterov:
            w_base = w + mu_col * v
            loss_base, grad_base = model.loss_and_grad(w_base, X_train, XT_train, D_train)
            if algo == "OptNAG":
                over = loss_base > loss_k
                if over.any():
                    v = torch.where(over[:, None], torch.zeros_like(v), v)
                    theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                    mu_col = torch.where(over[:, None], torch.zeros_like(mu_col), mu_col)
                    w_base = torch.where(over[:, None], w, w_base)
                    loss_base = torch.where(over, loss_k, loss_base)
                    grad_base = torch.where(over[:, None], grad_k, grad_base)
        else:
            w_base, loss_base, grad_base = w, loss_k, grad_k

        if use_armijo:
            g_raw = -grad_base
            scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = g_raw / scale
            slope = torch.sum(grad_base * g_k, dim=1)
            alpha = staged_parallel_armijo(
                model, w_base, g_k, loss_base, slope, X_train, D_train, chi=chi, tau=tau, ls_max=ls_max
            )
            v_next = mu_col * v + alpha[:, None] * g_k
        elif algo == "AdaGrad":
            r_state = r_state + grad_base * grad_base
            v_next = -lr * grad_base / torch.sqrt(r_state + 1e-8)
        elif algo == "RMSprop":
            theta_state = 0.9 * theta_state + 0.1 * (grad_base * grad_base)
            v_next = -lr * grad_base / torch.sqrt(theta_state + 1e-8)
        elif algo == "AdaDelta":
            theta_state = 0.95 * theta_state + 0.05 * (grad_base * grad_base)
            v_next = -(torch.sqrt(r_state + 1e-6) / torch.sqrt(theta_state + 1e-6)) * grad_base
            r_state = 0.95 * r_state + 0.05 * (v_next * v_next)
        elif algo == "Adam":
            b1, b2 = 0.9, 0.999
            r_state = b1 * r_state + (1.0 - b1) * grad_base
            theta_state = b2 * theta_state + (1.0 - b2) * (grad_base * grad_base)
            r_hat = r_state / (1.0 - (b1 ** k))
            th_hat = theta_state / (1.0 - (b2 ** k))
            v_next = -lr * r_hat / (torch.sqrt(th_hat) + 1e-8)
        else:
            raise ValueError(f"Unknown 1st-order algo: {algo}")

        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_train, XT_train, D_train)
        w, v, loss_k, grad_k = w_next, v_next, loss_next, grad_next

        if k % record_stride == 0:
            live_losses[orig_idx] = loss_k
            rec_epochs.append(k)
            rec_means.append(float(live_losses.mean().item()))

        if show_pbar and (k - last_k >= 2500):
            pbar.update(k - last_k)
            pbar.set_postfix(active=int(w.shape[0]), E_ave=f"{live_losses.mean().item()*1e3:.3f}")
            last_k = k

    if orig_idx.numel() > 0:
        gnorm = torch.norm(grad_k, dim=1)
        bad_num = (~torch.isfinite(loss_k)) | (~torch.isfinite(gnorm)) | (gnorm > 1e3)
        still = (gnorm > eps) & (~bad_num)
        if loss_stop is not None:
            still = still & (loss_k > loss_stop)
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
        loss_test = model.loss_only(w_final.to(X_test.dtype), X_test, D_test.to(X_test.dtype))

    e_tr = (loss_final.to(torch.float64) * 1000.0).cpu().numpy()
    e_te = (loss_test.to(torch.float64) * 1000.0).cpu().numpy()
    ep_np = epochs_final.cpu().numpy()
    div_np = diverged.cpu().numpy()
    if loss_stop is not None:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (loss_final.cpu().numpy() <= loss_stop))
    else:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (e_tr < 5.0))
    conv_rate = 100.0 * float(np.mean(conv_mask))
    ave_ep = float(ep_np.mean())
    per_iter_ms = (total_time / max(float(ep_np.sum()), 1.0)) * 1000.0
    est_time = (per_iter_ms / 1000.0) * ave_ep

    return {
        "algo": algo,
        "mu": f"{mu:.2f}" if algo in ("CM", "NAG") else "-",
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
        "rec_means": np.array(rec_means, dtype=np.float64),
        "best_weights": w_final[int(np.argmin(e_tr))].detach().clone(),
    }


def run_second_order_batched(
    algo: str,
    w_init: torch.Tensor,
    X_train: torch.Tensor,
    XT_train: torch.Tensor,
    D_train: torch.Tensor,
    X_test: torch.Tensor,
    D_test: torch.Tensor,
    n_in: int,
    n_hid: int,
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
    2次手法 (QN, MSQN, NAQ, AdaNAQ, MoQ, AdaMoQ) を完全倍精度 (IEEE 754 float64) で
    R 試行一括 GPU 並列学習するエンジン．
    さらに収束曲線の「反復回数軸」と「実計算時間軸」の双方を記録する．
    """
    assert algo in MoQFamilyOptimizer.VALID_ALGOS
    device = X_train.device
    dtype = torch.float64
    model = BatchedMLP(n_in=n_in, n_hid=n_hid, n_out=1)
    R, Dw = w_init.shape

    w = w_init.to(dtype).clone()
    v = torch.zeros_like(w)
    I_d = torch.eye(Dw, device=device, dtype=dtype)
    H = I_d.unsqueeze(0).expand(R, Dw, Dw).clone()
    theta_ada = torch.ones(R, device=device, dtype=dtype)

    w_final = w.clone()
    loss_final = torch.zeros(R, device=device, dtype=dtype)
    epochs_final = torch.zeros(R, device=device, dtype=torch.long)
    diverged = torch.zeros(R, device=device, dtype=torch.bool)
    orig_idx = torch.arange(R, device=device)

    is_moq = algo in ("MoQ", "AdaMoQ")
    is_naq = algo in ("NAQ", "AdaNAQ")
    is_ada = algo in ("AdaNAQ", "AdaMoQ")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_train, XT_train, D_train)
    grad_prev = grad_k.clone()
    w_prev = w.clone()
    s_hat_1 = torch.zeros_like(w)
    s_hat_2 = torch.zeros_like(w)

    live_losses = loss_k.clone()
    rec_epochs = [0]
    rec_times = [0.0]
    rec_means = [float(live_losses.mean().item())]
    rec_bests = [float(live_losses.min().item())]

    label = desc or (f"{algo} (mu={mu:.2f})" if algo in ("NAQ", "MoQ") else algo)
    pbar = tqdm(total=max_epochs, desc=label, disable=not show_pbar, mininterval=1.5)
    last_k = 0
    total_step_runs = 0

    for k in range(1, max_epochs + 1):
        if k % check_interval == 1:
            gnorm = torch.norm(grad_k, dim=1)
            bad_num = (~torch.isfinite(loss_k)) | (~torch.isfinite(gnorm)) | (gnorm > 1e3)
            conv = (gnorm <= eps) | bad_num
            if loss_stop is not None:
                conv = conv | ((loss_k <= loss_stop) & torch.isfinite(loss_k))
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
                H = H[alive]
                theta_ada = theta_ada[alive]
                loss_k = loss_k[alive]
                grad_k = grad_k[alive]
                grad_prev = grad_prev[alive]
                w_prev = w_prev[alive]
                s_hat_1 = s_hat_1[alive]
                s_hat_2 = s_hat_2[alive]

        Ra = w.shape[0]
        total_step_runs += Ra
        I_a = I_d.unsqueeze(0).expand(Ra, Dw, Dw)
        still = torch.norm(grad_k, dim=1) > eps
        if loss_stop is not None:
            still = still & (loss_k > loss_stop)

        # 1. mu_k の算出
        if algo in ("QN", "MSQN"):
            mu_vec = torch.zeros(Ra, device=device, dtype=dtype)
        elif is_ada:
            mu_vec, theta_ada = adaptive_momentum_step_tensor(theta_ada, gamma=gamma)
            mu_vec = torch.clamp(mu_vec, max=max_mu)
        else:
            mu_vec = torch.full((Ra,), mu, device=device, dtype=dtype)

        # 2. 基準点・基準損失・勾配（外挿勾配）の計算
        if algo in ("QN", "MSQN"):
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
                    if is_ada:
                        theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                    w_base = torch.where(over[:, None], w, w_base)
                    loss_base = torch.where(over, loss_k, loss_base)
                    grad_base = torch.where(over[:, None], grad_k, grad_base)
        else:  # MoQ / AdaMoQ: 損失のみ評価 + 線形外挿勾配
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
                    if is_ada:
                        theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                    w_base = torch.where(over[:, None], w, w_base)
                    loss_base = torch.where(over, loss_k, loss_base)
                    grad_base = torch.where(over[:, None], grad_k, grad_base)

        # 3. 探索方向 g_k = -H_k * grad_base
        g_raw = -torch.bmm(H, grad_base.unsqueeze(-1)).squeeze(-1)
        scale = torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
        g_k = g_raw / scale
        slope = torch.sum(grad_base * g_k, dim=1)

        bad = slope >= 0.0
        if bad.any():
            H = torch.where(bad[:, None, None], I_a, H)
            g_fb = -grad_base / torch.clamp(torch.norm(grad_base, dim=1, keepdim=True) / max_step_norm, min=1.0)
            g_k = torch.where(bad[:, None], g_fb, g_k)
            slope = torch.sum(grad_base * g_k, dim=1)

        # 4. 並列 Armijo 直線探索 (論文 Algorithm 10 準拠，最小ステップ tau**(ls_max-1))
        alpha = staged_parallel_armijo(
            model, w_base, g_k, loss_base, slope, X_train, D_train, chi=chi, tau=tau, ls_max=ls_max
        )

        # 5. 重み更新と更新後点の勾配計算
        v_next = mu_vec[:, None] * v + alpha[:, None] * g_k
        w_next = w + v_next
        loss_next, grad_next = model.loss_and_grad(w_next, X_train, XT_train, D_train)

        # 6. セカントベクトルとグローバル収束項（式(98)-(100)）による H_{k+1} 更新
        if algo == "MSQN":
            if k <= 2:
                nu_vec = torch.zeros(Ra, device=device, dtype=dtype)
            else:
                n1 = torch.norm(s_hat_1, dim=1)
                n2 = torch.norm(s_hat_2, dim=1)
                denom = 2.0 * (n2 * n2) + n1 * n2
                nu_vec = torch.where(denom > 1e-30, (n1 * n1) / torch.where(denom > 1e-30, denom, torch.ones_like(denom)), torch.zeros_like(denom))
            p_k = w_next - (1.0 + nu_vec[:, None]) * w + nu_vec[:, None] * w_prev
            delta = grad_next - (1.0 + nu_vec[:, None]) * grad_k + nu_vec[:, None] * grad_prev
            g_lin = grad_k
            s_hat_2 = torch.where(still[:, None], s_hat_1, s_hat_2)
            s_hat_1 = torch.where(still[:, None], p_k, s_hat_1)
        else:
            p_k = alpha[:, None] * g_k
            delta = grad_next - grad_base
            g_lin = grad_base

        if use_global_conv:
            p_norm_sq = torch.sum(p_k * p_k, dim=1)
            g_norm_sq = torch.sum(g_lin * g_lin, dim=1)
            g_norm = torch.sqrt(g_norm_sq)
            omega = torch.where(g_norm_sq > 1e-2, torch.full_like(g_norm_sq, 2.0), torch.full_like(g_norm_sq, 100.0)) * global_conv_scale
            dp = torch.sum(delta * p_k, dim=1)
            p_ns_safe = torch.where(p_norm_sq > 1e-30, p_norm_sq, torch.ones_like(p_norm_sq))
            xi_k = omega * g_norm + torch.clamp(-dp / p_ns_safe, min=0.0)
            q_k = torch.where((p_norm_sq > 1e-30)[:, None], delta + xi_k[:, None] * p_k, delta)
        else:
            q_k = delta

        pq = torch.sum(p_k * q_k, dim=1)
        Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
        qHq = torch.sum(q_k * Hq, dim=1)
        pq_s = torch.where(pq > curvature_eps, pq, torch.ones_like(pq))
        term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_s[:, None, None]
        term2 = (1.0 + qHq / pq_s)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_s[:, None, None]
        H_new = torch.where((pq > curvature_eps)[:, None, None], H - term1 + term2, I_a)

        w_prev = torch.where(still[:, None], w, w_prev)
        grad_prev = torch.where(still[:, None], grad_k, grad_prev)
        w = torch.where(still[:, None], w_next, w)
        v = torch.where(still[:, None], v_next, v)
        H = torch.where(still[:, None, None], H_new, H)
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
        still = (gnorm > eps) & (~bad_num)
        if loss_stop is not None:
            still = still & (loss_k > loss_stop)
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
    # 収束率: 発散せず，かつ規定反復以内または十分小さな誤差へ到達した割合
    if loss_stop is not None:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (loss_final.cpu().numpy() <= loss_stop))
    else:
        conv_mask = (~div_np) & ((ep_np < max_epochs) | (e_tr < 5.0))
    conv_rate = 100.0 * float(np.mean(conv_mask))
    ave_ep = float(ep_np.mean())
    per_iter_ms = (total_time / max(total_step_runs, 1)) * 1000.0
    est_time = (per_iter_ms / 1000.0) * ave_ep

    return {
        "algo": algo,
        "mu": f"{mu:.2f}" if algo in ("NAQ", "MoQ") else "-",
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


def format_moq_dataframe(results: List[Dict[str, object]], precision: int = 3) -> pd.DataFrame:
    """論文 表2 / 表7 / 表8 と同じ形式の pandas.DataFrame に整形する．"""
    fmt = f"{{:.{precision}f}}"
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r["algo"],
            "mu": r["mu"],
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
