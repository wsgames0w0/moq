# MoQ: 慣性付準ニュートン法に基づくニューラルネットワーク学習アルゴリズムの再現実験

博士論文「慣性付準ニュートン法に基づくニューラルネットワークの学習アルゴリズムに関する研究」（Shahrzad Mahboubi，湘南工科大学，2023 年 3 月）で提案された最適化アルゴリズム群を **Python / PyTorch（CUDA GPU 並列演算・倍精度 `torch.float64`）** で再現実装・評価したプロジェクトです．

---

## 実装アルゴリズム一覧

- **フルサイズ準ニュートン法（`ex001` / 第 2〜4 章）**：
  - ベースライン：`QN` (BFGS), `MSQN` ($m=2$), `NAQ` (固定 $\mu$), `AdaNAQ` (適応的 $\mu\_k$)
  - 提案法：**`MoQ`**（勾配線形外挿による先読み逆伝播削減）, **`AdaMoQ`**（適応的 $\mu\_k$ ＋ 勾配線形外挿）
  - 比較用 1 次手法：`GD`, `CM`, `NAG`, `OptNAG`, `AdaGrad`, `RMSprop`, `AdaDelta`, `Adam`
- **記憶制限準ニュートン法（`ex002` / 第 5 章，$\mathcal{O}(md)$ 空間・時間計算量）**：
  - `LQN`, **`LNAQ`**, **`LMoQ`**（Two-Loop Recursion 実装，$m \in \{4, 8, 16, 32\}$）
- **メモリレス準ニュートン法（`ex002` / 第 6 章，$\mathcal{O}(d)$ 空間・時間計算量）**：
  - `CG-HS` (Hestenes–Stiefel), `MLQN`, **`MLNAQ`**, **`MLMoQ`**（行列・履歴不要の閉形式内積更新）
- **パターン分類ベンチマーク（`ex003` / Softmax + 交差エントロピー誤差）**：
  - 上記全 14 手法による Three-Spirals・8×8・28×28 高次元手書き風数字分類の比較評価

---

## ディレクトリ構成

```text
moq/
├─ ex001/
│  ├─ model.py          # MLP モデル (FeedforwardMLP) & Flattened GEMM GPU バッチモデル (BatchedMLP)
│  ├─ utils.py          # MoQFamilyOptimizer, Beale 軌跡, 1次/2次並列エンジン, 2D f1・5D Levy ローダ
│  └─ ex001.ipynb       # 実験 1 実行済みノートブック (数値等価性, Beale, f1 2D, Levy 5D)
├─ ex002/
│  ├─ model.py          # 記憶制限・メモリレス実験用 MLP & TorchScript バッチモデル
│  ├─ utils.py          # LimitedMemoryMoQOptimizer, MemoryLessMoQOptimizer, 理論計算量解析
│  └─ ex002.ipynb       # 実験 2 実行済みノートブック (等価性, 理論スケーリング, 1D f(x), Levy 5D)
├─ ex003/
│  ├─ model.py          # 多クラス分類用 MLP & BatchedClassifier (Softmax + CE)
│  ├─ utils.py          # Three-Spirals・8x8/28x28 数字画像生成, 14手法並列分類エンジン
│  └─ ex003.ipynb       # 実験 3 実行済みノートブック (Spirals, 8x8 Digits, 28x28 High-Dim Digits)
├─ report/
│  ├─ img/              # 実験・解析レポート図表 (9 枚)
│  ├─ ex001.md          # 実験 1 詳細レポート
│  ├─ ex002.md          # 実験 2 詳細レポート
│  ├─ ex003.md          # 実験 3 詳細レポート
│  └─ report.md         # 全体総括レポート（理論・誤植検証・Python/PyTorch 実装詳細解説）
├─ datasets/            # 自動生成・キャッシュデータセットおよび結果 CSV（Git 管理対象外）
├─ requirements.txt
├─ LICENSE              # MIT License
└─ .gitignore
```

---

## 環境構築と実行方法

- **動作確認環境**：Python 3.10.11 / PyTorch `2.7.1+cu118` / NVIDIA GeForce RTX 4080（倍精度 `torch.float64`）

```powershell
# 1. 仮想環境の作成と有効化
python -m venv venv
.\venv\Scripts\activate

# 2. GPU 対応版 PyTorch (CUDA 11.8) および依存ライブラリのインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

各実験ディレクトリ（`ex001/`, `ex002/`, `ex003/`）に移動して Jupyter Notebook（`ex001.ipynb`, `ex002.ipynb`, `ex003.ipynb`）を先頭から順に実行すると，データセット（`datasets/`）・結果 CSV・レポート用図表（`report/img/`）が生成されます．

---

## 主な再現結果と技術的ポイント

1. **Optimizer と GPU 並列バッチエンジンの厳密な等価性**：
   標準的な PyTorch `torch.optim.Optimizer` 継承クラス（`MoQFamilyOptimizer`, `LimitedMemoryMoQOptimizer`, `MemoryLessMoQOptimizer`）と Flattened GEMM GPU 並列実行エンジンを実装し，倍精度で **最大差分 $\le 4.45 \times 10^{-14}$**（収束後期の `MLQN` を含む全 13 手法で一致）を検証した．
2. **計算コスト削減の実証（`MoQ` / `AdaMoQ` / `LMoQ` / `MLMoQ`）**：
   先読み点における勾配を $(1+\mu\_k)\nabla E(\mathbf{w}\_k) - \mu\_k\nabla E(\mathbf{w}\_{k-1})$ の線形外挿で近似することで誤差逆伝播を 1 回に半減させ，実測時間でも `NAQ` 系に対し **19〜28% 高速** な 1 反復実行時間を達成した．
3. **非線形関数近似・高次元分類性能**：
   - **Beale 関数**：全 2 次手法が 14〜16 反復で最適解 $(3.0, 0.5)$（$f \sim 10^{-19}\sim 10^{-15}$）へ到達．
   - **2D 関数 $f\_1(x, a)$**：`AdaMoQ` が 1 次手法（`Adam` 中央値 `28.957`）や `QN`/`MSQN` を大きく上回る訓練誤差 `3.840` を達成．
   - **1D 関数 $f(x)$**：`LMoQ (m=32)` が平均誤差 `2.152`（`LQN` の約 1/2 以下）かつ最悪試行値 `6.472` の高い安定性を実証．
   - **Three-Spirals 分類**：全 2 次手法が訓練正解率 100.00% を達成し，`MLMoQ` がテスト精度中央値 100.00%・平均 99.98% を記録．
   - **28×28 高次元数字分類（$d=7,960$）**：`Adam`（テスト中央値 `55.83%`）や `CG`（`77.80%`）を大きく上回り，**`LMoQ (m=8)` が最高訓練精度（中央値 `86.05%`・最良 `94.96%`）**，**`MLMoQ` が最高テスト精度（中央値 `81.15%`・平均 `82.00%`・最良 `87.55%`）** を達成．

詳細なアルゴリズム解説・論文中数式の精査結果・PyTorch 実装コード解説は **[全体レポート (report/report.md)](./report/report.md)** および個別レポート（[ex001.md](./report/ex001.md) / [ex002.md](./report/ex002.md) / [ex003.md](./report/ex003.md)）を参照してください．

---

## ライセンス

本プロジェクトは [MIT License](./LICENSE) の下で公開されています．
