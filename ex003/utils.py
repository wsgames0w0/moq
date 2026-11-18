"""
ex003/utils.py
博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」
(Shahrzad Mahboubi, 2023) の実験3（分類問題ベンチマーク: Three-Spirals, 8x8 MNIST, 28x28 MNIST風分類）
用データセット生成・オプティマイザ・バッチ並列学習エンジン．
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.optim import Optimizer
from tqdm import tqdm

from model import BatchedClassifier


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


# =====================================================================
# 分類用データセット生成・読込 (Three-Spirals, 8x8 Digits, 28x28 Digits)
# =====================================================================

def load_or_create_three_spirals_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    論文 4.3.6節・6.5.2節 Three-Spirals データセット (図49):
      3クラス x 350サンプル = 1,050 学習データ + 同分布から独立生成した 1,050 テストデータ．
      座標を [-1.0, 1.0] の範囲に収めて返す．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_three_spirals.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_tr, y_tr = d["X_tr"], d["y_tr"]
        X_te, y_te = d["X_te"], d["y_te"]
    else:
        rng = np.random.default_rng(seed)

        def _gen_spirals(n_per_class: int):
            xs, ys = [], []
            for c in range(3):
                r = np.linspace(0.06, 0.96, n_per_class)
                theta = (c * 2.0 * np.pi / 3.0) + np.linspace(0.0, 1.25 * np.pi, n_per_class)
                theta += rng.normal(0.0, 0.08, size=n_per_class)
                r_noisy = np.clip(r + rng.normal(0.0, 0.02, size=n_per_class), 0.02, 0.99)
                px = r_noisy * np.cos(theta)
                py = r_noisy * np.sin(theta)
                xs.append(np.stack([px, py], axis=1))
                ys.append(np.full(n_per_class, c, dtype=np.int64))
            X_all = np.concatenate(xs, axis=0)
            y_all = np.concatenate(ys, axis=0)
            perm = rng.permutation(len(y_all))
            return X_all[perm].astype(np.float64), y_all[perm]

        X_tr, y_tr = _gen_spirals(350)
        X_te, y_te = _gen_spirals(350)
        np.savez(npz_path, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(X_tr, device=device, dtype=torch.float64)
    y_tr_t = torch.tensor(y_tr, device=device, dtype=torch.long)
    D_tr_oh = torch.nn.functional.one_hot(y_tr_t, num_classes=3).to(torch.float64)

    X_te_t = torch.tensor(X_te, device=device, dtype=torch.float64)
    y_te_t = torch.tensor(y_te, device=device, dtype=torch.long)
    D_te_oh = torch.nn.functional.one_hot(y_te_t, num_classes=3).to(torch.float64)

    return {
        "X_train_64": X_tr_t,
        "XT_train_64": X_tr_t.t().contiguous(),
        "D_train_64": D_tr_oh,
        "y_train": y_tr_t,
        "X_test_64": X_te_t,
        "D_test_64": D_te_oh,
        "y_test": y_te_t,
        "X_tr_raw": X_tr,
        "y_tr_raw": y_tr,
    }


def _find_digit_fonts() -> List[str]:
    """Windows 標準フォントディレクトリから数字描画用 TTF フォントパス一覧を取得する．"""
    font_dir = r"C:\Windows\Fonts"
    candidates = [
        "segoepr.ttf", "segoeprb.ttf", "comic.ttf", "comicbd.ttf", "comici.ttf",
        "arial.ttf", "ariali.ttf", "arialbd.ttf", "times.ttf", "timesi.ttf",
        "cour.ttf", "couri.ttf", "georgia.ttf", "calibri.ttf", "calibrii.ttf",
        "tahoma.ttf", "verdana.ttf", "trebuc.ttf",
    ]
    found = [os.path.join(font_dir, c) for c in candidates if os.path.exists(os.path.join(font_dir, c))]
    return found


def _generate_handwritten_style_digits(
    num_samples: int, img_size: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    複数フォント＋アフィン歪み (回転・せん断・平行移動・異方スケール)・平滑化・
    微小画素揺らぎを組み合わせて手書き数字風の画像データセットを生成する．
    """
    rng = np.random.default_rng(seed)
    font_paths = _find_digit_fonts()
    fonts_loaded = []
    canvas_sz = 48
    for fp in font_paths:
        for fsz in (28, 32, 35):
            try:
                fonts_loaded.append(ImageFont.truetype(fp, fsz))
            except Exception:
                pass
    if not fonts_loaded:
        fonts_loaded = [ImageFont.load_default()]

    images = np.zeros((num_samples, img_size * img_size), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for idx in range(num_samples):
        digit = idx % 10
        labels[idx] = digit
        font = fonts_loaded[int(rng.integers(0, len(fonts_loaded)))]

        img = Image.new("L", (canvas_sz, canvas_sz), 0)
        draw = ImageDraw.Draw(img)
        txt = str(digit)
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = (canvas_sz - tw) / 2.0 - bbox[0] + float(rng.uniform(-3.0, 3.0))
        oy = (canvas_sz - th) / 2.0 - bbox[1] + float(rng.uniform(-3.0, 3.0))
        draw.text((ox, oy), txt, fill=int(rng.integers(205, 255)), font=font)

        # アフィン変形 (回転 + せん断 + 異方性スケール)
        angle = float(rng.uniform(-18.0, 18.0))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        shear_x = float(rng.uniform(-0.18, 0.18))
        shear_y = float(rng.uniform(-0.10, 0.10))
        sx = float(rng.uniform(0.86, 1.14))
        sy = float(rng.uniform(0.88, 1.12))
        cx, cy = canvas_sz / 2.0, canvas_sz / 2.0
        # 中心基準の変換行列
        a, b = sx, shear_x
        d, e = shear_y, sy
        c = cx - a * cx - b * cy
        f = cy - d * cx - e * cy
        img = img.transform(
            (canvas_sz, canvas_sz),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

        # 太さ揺らぎ（ガウシアンブラー）
        blur_r = float(rng.uniform(0.4, 1.3))
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # 目標解像度へ縮小
        img_small = img.resize((img_size, img_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img_small, dtype=np.float32) / 255.0
        max_v = float(arr.max())
        if max_v > 1e-4:
            arr = arr / max_v
        # 微小な背景・ストローク揺らぎ
        arr = np.clip(arr + rng.normal(0.0, 0.03, size=arr.shape).astype(np.float32), 0.0, 1.0)
        images[idx] = arr.reshape(-1)

    perm = rng.permutation(num_samples)
    return images[perm], labels[perm]


def load_or_create_digits_8x8_dataset(
    dataset_dir: str = "../datasets",
    seed: int = 2023,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    論文 4.3.5節・6.5.3節 8x8 手書き数字文字分類問題:
      全 1,797 サンプル (75% = 1,347 訓練 / 25% = 450 テスト)，64次元，10クラス．
      入力ピクセル値は [-1.0, 1.0] に正規化．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_digits_8x8.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_tr, y_tr = d["X_tr"], d["y_tr"]
        X_te, y_te = d["X_te"], d["y_te"]
    else:
        X_all, y_all = _generate_handwritten_style_digits(1797, img_size=8, seed=seed)
        X_all_norm = (2.0 * X_all - 1.0).astype(np.float64)
        X_tr, y_tr = X_all_norm[:1347], y_all[:1347]
        X_te, y_te = X_all_norm[1347:], y_all[1347:]
        np.savez(npz_path, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(X_tr, device=device, dtype=torch.float64)
    y_tr_t = torch.tensor(y_tr, device=device, dtype=torch.long)
    D_tr_oh = torch.nn.functional.one_hot(y_tr_t, num_classes=10).to(torch.float64)

    X_te_t = torch.tensor(X_te, device=device, dtype=torch.float64)
    y_te_t = torch.tensor(y_te, device=device, dtype=torch.long)
    D_te_oh = torch.nn.functional.one_hot(y_te_t, num_classes=10).to(torch.float64)

    return {
        "X_train_64": X_tr_t,
        "XT_train_64": X_tr_t.t().contiguous(),
        "D_train_64": D_tr_oh,
        "y_train": y_tr_t,
        "X_test_64": X_te_t,
        "D_test_64": D_te_oh,
        "y_test": y_te_t,
        "X_tr_img": X_tr,
        "y_tr_raw": y_tr,
    }


def load_or_create_digits_28x28_dataset(
    dataset_dir: str = "../datasets",
    n_train: int = 12000,
    n_test: int = 2000,
    seed: int = 2024,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    論文 6.5.4節 28x28 数字画像分類問題 (784-10-10, dim(w)=7,960):
      フルサイズ逆ヘッセ行列 (7960x7960) の保持が非現実的な大規模次元での
      Adam / MLQN / MLNAQ / MLMoQ / LMoQ(m=8) の性能比較用データセット．
    """
    os.makedirs(dataset_dir, exist_ok=True)
    npz_path = os.path.join(dataset_dir, "moq_digits_28x28.npz")

    if os.path.exists(npz_path):
        d = np.load(npz_path)
        X_tr, y_tr = d["X_tr"], d["y_tr"]
        X_te, y_te = d["X_te"], d["y_te"]
    else:
        X_all, y_all = _generate_handwritten_style_digits(n_train + n_test, img_size=28, seed=seed)
        X_all_norm = (2.0 * X_all - 1.0).astype(np.float32)
        X_tr, y_tr = X_all_norm[:n_train], y_all[:n_train]
        X_te, y_te = X_all_norm[n_train:], y_all[n_train:]
        np.savez(npz_path, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(X_tr, device=device, dtype=torch.float64)
    y_tr_t = torch.tensor(y_tr, device=device, dtype=torch.long)
    D_tr_oh = torch.nn.functional.one_hot(y_tr_t, num_classes=10).to(torch.float64)

    X_te_t = torch.tensor(X_te, device=device, dtype=torch.float64)
    y_te_t = torch.tensor(y_te, device=device, dtype=torch.long)
    D_te_oh = torch.nn.functional.one_hot(y_te_t, num_classes=10).to(torch.float64)

    return {
        "X_train_64": X_tr_t,
        "XT_train_64": X_tr_t.t().contiguous(),
        "D_train_64": D_tr_oh,
        "y_train": y_tr_t,
        "X_test_64": X_te_t,
        "D_test_64": D_te_oh,
        "y_test": y_te_t,
        "X_tr_img": X_tr,
        "y_tr_raw": y_tr,
    }


def generate_initial_weights(
    num_runs: int, num_params: int, seed: int = 2023, device: Optional[torch.device] = None
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.rand((num_runs, num_params), generator=gen, dtype=torch.float64) - 0.5).to(device)


def staged_parallel_armijo_cls(
    model: BatchedClassifier,
    w_base: torch.Tensor,
    g_k: torch.Tensor,
    loss_base: torch.Tensor,
    slope: torch.Tensor,
    X: torch.Tensor,
    D_onehot: torch.Tensor,
    chi: float = 1e-3,
    tau: float = 0.5,
    ls_max: int = 10,
) -> torch.Tensor:
    Ra, Dw = w_base.shape
    alpha = torch.ones(Ra, device=w_base.device, dtype=w_base.dtype)
    l0 = model.loss_only(w_base + g_k, X, D_onehot)
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
        lc = model.loss_only(wc.reshape(rem.numel() * K, Dw), X, D_onehot).view(rem.numel(), K)
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


def _batched_two_loop_cls(
    grad_base: torch.Tensor,
    p_hist: List[torch.Tensor],
    q_hist: List[torch.Tensor],
    pq_hist: List[torch.Tensor],
    valid_hist: List[torch.Tensor],
) -> torch.Tensor:
    g = -grad_base.clone()
    n_hist = len(p_hist)
    if n_hist == 0:
        return g
    alphas: List[torch.Tensor] = [torch.zeros(grad_base.shape[0], device=grad_base.device, dtype=grad_base.dtype)] * n_hist
    for i in range(n_hist - 1, -1, -1):
        v_mask = valid_hist[i]
        pq_s = torch.where(v_mask, pq_hist[i], torch.ones_like(pq_hist[i]))
        phi_i = torch.where(v_mask, torch.sum(p_hist[i] * g, dim=1) / pq_s, torch.zeros_like(pq_s))
        alphas[i] = phi_i
        g = g - phi_i[:, None] * q_hist[i]

    v_last = valid_hist[-1]
    qq_last = torch.sum(q_hist[-1] * q_hist[-1], dim=1)
    safe_qq = torch.where(v_last & (qq_last > 1e-30), qq_last, torch.ones_like(qq_last))
    gamma_k = torch.where(v_last & (qq_last > 1e-30), pq_hist[-1] / safe_qq, torch.ones_like(qq_last))
    g = g * gamma_k[:, None]

    for i in range(n_hist):
        v_mask = valid_hist[i]
        pq_s = torch.where(v_mask, pq_hist[i], torch.ones_like(pq_hist[i]))
        tau = torch.where(v_mask, torch.sum(q_hist[i] * g, dim=1) / pq_s, torch.zeros_like(pq_s))
        g = g + (alphas[i] - tau)[:, None] * p_hist[i]
    return g


def run_classification_batched(
    algo: str,
    w_init: torch.Tensor,
    ds: Dict[str, torch.Tensor],
    n_in: int,
    n_hid: int,
    n_out: int,
    memory_size: int = 8,
    lr_adam: float = 0.01,
    chi: float = 1e-3,
    tau: float = 0.5,
    ls_max: int = 10,
    max_step_norm: float = 20.0,
    curvature_eps: float = 1e-14,
    max_mu: float = 0.92,
    eps: float = 1e-6,
    loss_stop: Optional[float] = None,
    max_epochs: int = 10000,
    fixed_stepsize_1_0: bool = False,
    check_interval: int = 50,
    record_stride: int = 20,
    show_pbar: bool = True,
    desc: Optional[str] = None,
) -> Dict[str, object]:
    """
    分類問題用 R 試行一括 GPU 並列学習エンジン．
    全手法 (Adam, CG, QN, MSQN, NAQ, AdaNAQ, MoQ, AdaMoQ, LQN, LNAQ, LMoQ, MLQN, MLNAQ, MLMoQ) をサポートする．
    """
    valid_algos = (
        "Adam", "CG", "QN", "MSQN", "NAQ", "AdaNAQ", "MoQ", "AdaMoQ",
        "LQN", "LNAQ", "LMoQ", "MLQN", "MLNAQ", "MLMoQ",
    )
    assert algo in valid_algos, f"Unsupported algo {algo}"

    X_tr, XT_tr, D_tr, y_tr = ds["X_train_64"], ds["XT_train_64"], ds["D_train_64"], ds["y_train"]
    X_te, D_te, y_te = ds["X_test_64"], ds["D_test_64"], ds["y_test"]
    device = X_tr.device
    dtype = torch.float64
    model = BatchedClassifier(n_in=n_in, n_hid=n_hid, n_out=n_out)
    R, Dw = w_init.shape

    w = w_init.to(dtype).clone()
    v = torch.zeros_like(w)
    theta_ada = torch.ones(R, device=device, dtype=dtype)

    is_full_qn = algo in ("QN", "MSQN", "NAQ", "AdaNAQ", "MoQ", "AdaMoQ")
    is_limited = algo in ("LQN", "LNAQ", "LMoQ")
    is_ml = algo in ("MLQN", "MLNAQ", "MLMoQ")
    is_cg = algo == "CG"
    is_adam = algo == "Adam"
    is_moq = algo in ("MoQ", "AdaMoQ", "LMoQ", "MLMoQ")
    is_naq = algo in ("NAQ", "AdaNAQ", "LNAQ", "MLNAQ")
    is_ada = algo in ("AdaNAQ", "AdaMoQ", "LNAQ", "LMoQ", "MLNAQ", "MLMoQ")

    I_d = torch.eye(Dw, device=device, dtype=dtype) if is_full_qn else None
    H = I_d.unsqueeze(0).expand(R, Dw, Dw).clone() if is_full_qn else None

    p_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    q_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    pq_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)
    valid_hist: Deque[torch.Tensor] = deque(maxlen=memory_size)

    p_prev = torch.zeros_like(w) if is_ml else None
    q_prev = torch.zeros_like(w) if is_ml else None
    pq_prev = torch.zeros(R, device=device, dtype=dtype) if is_ml else None
    qq_prev = torch.zeros(R, device=device, dtype=dtype) if is_ml else None
    has_prev = torch.zeros(R, device=device, dtype=torch.bool)

    g_cg_prev = torch.zeros_like(w) if is_cg else None
    y_cg_prev = torch.zeros_like(w) if is_cg else None

    r_adam = torch.zeros_like(w) if is_adam else None
    th_adam = torch.zeros_like(w) if is_adam else None

    w_final = w.clone()
    loss_final = torch.zeros(R, device=device, dtype=dtype)
    epochs_final = torch.zeros(R, device=device, dtype=torch.long)
    diverged = torch.zeros(R, device=device, dtype=torch.bool)
    orig_idx = torch.arange(R, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    loss_k, grad_k = model.loss_and_grad(w, X_tr, XT_tr, D_tr)
    grad_prev = grad_k.clone()
    w_prev = w.clone()
    s_hat_1 = torch.zeros_like(w)
    s_hat_2 = torch.zeros_like(w)

    live_losses = loss_k.clone()
    with torch.no_grad():
        live_tr_acc = model.accuracy(w, X_tr, y_tr)
        live_te_acc = model.accuracy(w, X_te, y_te)

    rec_epochs = [0]
    rec_times = [0.0]
    rec_means = [float(live_losses.mean().item())]
    rec_bests = [float(live_losses.min().item())]
    rec_tr_accs = [float(live_tr_acc.mean().item())]
    rec_te_accs = [float(live_te_acc.mean().item())]

    label = desc or (f"{algo}(m={memory_size})" if is_limited else algo)
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
                with torch.no_grad():
                    live_tr_acc[d_idx] = model.accuracy(w[conv], X_tr, y_tr)
                    live_te_acc[d_idx] = model.accuracy(w[conv], X_te, y_te)
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
                w_prev = w_prev[alive]
                s_hat_1 = s_hat_1[alive]
                s_hat_2 = s_hat_2[alive]
                has_prev = has_prev[alive]
                if H is not None:
                    H = H[alive]
                if is_limited:
                    p_hist = deque([t[alive] for t in p_hist], maxlen=memory_size)
                    q_hist = deque([t[alive] for t in q_hist], maxlen=memory_size)
                    pq_hist = deque([t[alive] for t in pq_hist], maxlen=memory_size)
                    valid_hist = deque([t[alive] for t in valid_hist], maxlen=memory_size)
                if is_ml:
                    p_prev = p_prev[alive]
                    q_prev = q_prev[alive]
                    pq_prev = pq_prev[alive]
                    qq_prev = qq_prev[alive]
                if is_cg:
                    g_cg_prev = g_cg_prev[alive]
                    y_cg_prev = y_cg_prev[alive]
                if is_adam:
                    r_adam = r_adam[alive]
                    th_adam = th_adam[alive]

        Ra = w.shape[0]
        total_step_runs += Ra
        still = torch.norm(grad_k, dim=1) > eps
        if loss_stop is not None:
            still = still & (loss_k > loss_stop)

        if is_adam:
            b1, b2 = 0.9, 0.999
            r_adam = b1 * r_adam + (1.0 - b1) * grad_k
            th_adam = b2 * th_adam + (1.0 - b2) * (grad_k * grad_k)
            r_hat = r_adam / (1.0 - (b1 ** k))
            th_hat = th_adam / (1.0 - (b2 ** k))
            v_next = -lr_adam * r_hat / (torch.sqrt(th_hat) + 1e-8)
            w_next = w + v_next
            loss_next, grad_next = model.loss_and_grad(w_next, X_tr, XT_tr, D_tr)
            w = torch.where(still[:, None], w_next, w)
            v = torch.where(still[:, None], v_next, v)
            loss_k = torch.where(still, loss_next, loss_k)
            grad_k = torch.where(still[:, None], grad_next, grad_k)
        else:
            if algo in ("QN", "MSQN", "LQN", "CG", "MLQN"):
                mu_vec = torch.zeros(Ra, device=device, dtype=dtype)
            elif is_ada:
                mu_vec, theta_ada = adaptive_momentum_step_tensor(theta_ada, gamma=1e-5)
                mu_vec = torch.clamp(mu_vec, max=max_mu)
            else:
                mu_vec = torch.full((Ra,), 0.85, device=device, dtype=dtype)

            if algo in ("QN", "MSQN", "LQN", "CG", "MLQN"):
                w_base, loss_base, grad_base = w, loss_k, grad_k
            elif is_naq:
                if float(mu_vec.max().item()) == 0.0:
                    w_base, loss_base, grad_base = w, loss_k, grad_k
                else:
                    w_base = w + mu_vec[:, None] * v
                    loss_base, grad_base = model.loss_and_grad(w_base, X_tr, XT_tr, D_tr)
                    over = loss_base > loss_k
                    if over.any():
                        v = torch.where(over[:, None], torch.zeros_like(v), v)
                        mu_vec = torch.where(over, torch.zeros_like(mu_vec), mu_vec)
                        if is_ada:
                            theta_ada = torch.where(over, torch.ones_like(theta_ada), theta_ada)
                        w_base = torch.where(over[:, None], w, w_base)
                        loss_base = torch.where(over, loss_k, loss_base)
                        grad_base = torch.where(over[:, None], grad_k, grad_base)
            else:  # MoQ系
                if k == 1 or float(mu_vec.max().item()) == 0.0:
                    w_base, loss_base, grad_base = w, loss_k, grad_k
                else:
                    w_base = w + mu_vec[:, None] * v
                    loss_base = model.loss_only(w_base, X_tr, D_tr)
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

            I_a = I_d.unsqueeze(0).expand(Ra, Dw, Dw) if is_full_qn else None

            if is_full_qn:
                g_raw = -torch.bmm(H, grad_base.unsqueeze(-1)).squeeze(-1)
            elif is_limited:
                g_raw = _batched_two_loop_cls(
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
                bk = torch.sum(p_prev * grad_base, dim=1) / pq_s
                pk_s = torch.sum(q_prev * grad_base, dim=1) / pq_s
                ok = qq_s / pq_s
                tk = pq_s / qq_s
                g_ml = -tk[:, None] * (grad_base - (bk[:, None] * q_prev + pk_s[:, None] * p_prev)) - ((1.0 + tk * ok) * bk)[:, None] * p_prev
                g_raw = torch.where(use_prev[:, None], g_ml, -grad_base)

            g_k = g_raw / torch.clamp(torch.norm(g_raw, dim=1, keepdim=True) / max_step_norm, min=1.0)
            slope = torch.sum(grad_base * g_k, dim=1)
            bad = slope >= 0.0
            if bad.any():
                if is_full_qn:
                    H = torch.where(bad[:, None, None], I_a, H)
                if is_ml or is_cg:
                    has_prev = has_prev & (~bad)
                g_fb = -grad_base / torch.clamp(torch.norm(grad_base, dim=1, keepdim=True) / max_step_norm, min=1.0)
                g_k = torch.where(bad[:, None], g_fb, g_k)
                slope = torch.sum(grad_base * g_k, dim=1)

            if fixed_stepsize_1_0:
                alpha = torch.ones(Ra, device=device, dtype=dtype)
            else:
                alpha = staged_parallel_armijo_cls(
                    model, w_base, g_k, loss_base, slope, X_tr, D_tr, chi=chi, tau=tau, ls_max=ls_max
                )

            v_next = mu_vec[:, None] * v + alpha[:, None] * g_k
            w_next = w + v_next
            loss_next, grad_next = model.loss_and_grad(w_next, X_tr, XT_tr, D_tr)

            if is_cg:
                g_cg_prev = torch.where(still[:, None], g_k, g_cg_prev)
                y_cg_prev = torch.where(still[:, None], grad_next - grad_k, y_cg_prev)
                has_prev = has_prev | still
            else:
                if algo == "MSQN":
                    if k <= 2:
                        nu_vec = torch.zeros(Ra, device=device, dtype=dtype)
                    else:
                        n1 = torch.norm(s_hat_1, dim=1)
                        n2 = torch.norm(s_hat_2, dim=1)
                        denom = 2.0 * (n2 * n2) + n1 * n2
                        nu_vec = torch.where(denom > 1e-30, (n1 * n1) / torch.where(denom > 1e-30, denom, torch.ones_like(denom)), torch.zeros_like(denom))
                    p_k = w_next - (1.0 + nu_vec[:, None]) * w + nu_vec[:, None] * w_prev
                    q_k = grad_next - (1.0 + nu_vec[:, None]) * grad_k + nu_vec[:, None] * grad_prev
                    s_hat_2 = torch.where(still[:, None], s_hat_1, s_hat_2)
                    s_hat_1 = torch.where(still[:, None], p_k, s_hat_1)
                else:
                    p_k = alpha[:, None] * g_k
                    q_k = grad_next - grad_base

                pq = torch.sum(p_k * q_k, dim=1)
                qq = torch.sum(q_k * q_k, dim=1)
                ok_pair = still & (pq > curvature_eps) & (qq > 1e-30)

                if is_full_qn:
                    Hq = torch.bmm(H, q_k.unsqueeze(-1)).squeeze(-1)
                    qHq = torch.sum(q_k * Hq, dim=1)
                    pq_s = torch.where(pq > curvature_eps, pq, torch.ones_like(pq))
                    term1 = (Hq.unsqueeze(2) * p_k.unsqueeze(1) + p_k.unsqueeze(2) * Hq.unsqueeze(1)) / pq_s[:, None, None]
                    term2 = (1.0 + qHq / pq_s)[:, None, None] * (p_k.unsqueeze(2) * p_k.unsqueeze(1)) / pq_s[:, None, None]
                    H_new = torch.where((pq > curvature_eps)[:, None, None], H - term1 + term2, I_a)
                    H = torch.where(still[:, None, None], H_new, H)
                elif is_limited:
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

            w_prev = torch.where(still[:, None], w, w_prev)
            grad_prev = torch.where(still[:, None], grad_k, grad_prev)
            w = torch.where(still[:, None], w_next, w)
            v = torch.where(still[:, None], v_next, v)
            loss_k = torch.where(still, loss_next, loss_k)
            grad_k = torch.where(still[:, None], grad_next, grad_k)

        if k % record_stride == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            live_losses[orig_idx] = loss_k
            with torch.no_grad():
                live_tr_acc[orig_idx] = model.accuracy(w, X_tr, y_tr)
                live_te_acc[orig_idx] = model.accuracy(w, X_te, y_te)
            rec_epochs.append(k)
            rec_times.append(time.perf_counter() - t0)
            rec_means.append(float(live_losses.mean().item()))
            rec_bests.append(float(live_losses.min().item()))
            rec_tr_accs.append(float(live_tr_acc.mean().item()))
            rec_te_accs.append(float(live_te_acc.mean().item()))

        if show_pbar and (k - last_k >= max(100, max_epochs // 20)):
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
        loss_test = model.loss_only(w_final, X_te, D_te)
        acc_tr = model.accuracy(w_final, X_tr, y_tr)
        acc_te = model.accuracy(w_final, X_te, y_te)

    e_tr = (loss_final * 1000.0).cpu().numpy()
    e_te = (loss_test * 1000.0).cpu().numpy()
    acc_tr_np = acc_tr.cpu().numpy()
    acc_te_np = acc_te.cpu().numpy()
    ep_np = epochs_final.cpu().numpy()
    div_np = diverged.cpu().numpy()
    conv_rate = 100.0 * float(np.mean(~div_np))
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
        "acc_train_median": float(np.median(acc_tr_np)),
        "acc_train_ave": float(np.mean(acc_tr_np)),
        "acc_train_best": float(np.max(acc_tr_np)),
        "acc_train_worst": float(np.min(acc_tr_np)),
        "acc_test_median": float(np.median(acc_te_np)),
        "acc_test_ave": float(np.mean(acc_te_np)),
        "acc_test_best": float(np.max(acc_te_np)),
        "acc_test_worst": float(np.min(acc_te_np)),
        "ave_epochs": ave_ep,
        "ave_time_sec": est_time,
        "per_iter_ms": per_iter_ms,
        "batch_time_sec": total_time,
        "conv_rate": conv_rate,
        "rec_epochs": np.array(rec_epochs, dtype=np.int64),
        "rec_times": np.array(rec_times, dtype=np.float64),
        "rec_means": np.array(rec_means, dtype=np.float64),
        "rec_bests": np.array(rec_bests, dtype=np.float64),
        "rec_tr_accs": np.array(rec_tr_accs, dtype=np.float64),
        "rec_te_accs": np.array(rec_te_accs, dtype=np.float64),
        "best_weights": w_final[int(np.argmin(e_tr))].detach().clone(),
    }


def format_cls_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Algorithm": r["algo"],
            "m": r["m"],
            "E_train (x10^3) Med/Ave/Best/Worst": f"{r['train_median']:.3f} / {r['train_ave']:.3f} / {r['train_best']:.3f} / {r['train_worst']:.3f}",
            "Est. Time(s)": round(r["ave_time_sec"], 3),
            "Per Iter(ms)": round(r["per_iter_ms"], 3),
            "Ave. Iters": int(round(r["ave_epochs"])),
            "Acc_train(%) Med/Ave/Best/Worst": f"{r['acc_train_median']:.2f} / {r['acc_train_ave']:.2f} / {r['acc_train_best']:.2f} / {r['acc_train_worst']:.2f}",
            "Acc_test(%) Med/Ave/Best/Worst": f"{r['acc_test_median']:.2f} / {r['acc_test_ave']:.2f} / {r['acc_test_best']:.2f} / {r['acc_test_worst']:.2f}",
            "Conv(%)": round(r["conv_rate"], 1),
        })
    return pd.DataFrame(rows)
