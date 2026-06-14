# Boolean Network Edge Removal & Q-Learning

このリポジトリは、ブーリアンネットワークの安定化制御におけるエッジ除去戦略を強化学習（Q-Learning）を用いて最適化する研究の公開用コードベースです。

## 特徴
- **高速なシミュレーション**: Pythonによるブーリアンネットワーク状態遷移の効率的な計算。
- **再現性**: `uv` と `poethepoet` を活用したコマンド一発での環境構築と実験実行。
- **インタラクティブな分析**: `marimo` ノートブックを用いた、成果の可視化とWebダッシュボード。

## クイックスタート (ローカル実行)

本リポジトリは `uv` パッケージマネージャを使用しています。事前に[インストール](https://github.com/astral-sh/uv)を行ってください。

### 1. リポジトリのクローンと環境構築
```bash
git clone https://github.com/koshikai/bn-edge-removal-public.git
cd bn-edge-removal-public
uv sync
```

### 2. ダッシュボードの起動（Marimo）
インタラクティブな成果確認のため、各種ダッシュボードを用意しています。

```bash
# 1. ネットワーク構造の可視化（Cortical, Wnt5aなど）
uv run poe net

# 2. 実験結果（学習曲線・状態軌道）の可視化
# 注意: 事前に実験(poe exp など)を実行し、outputs/ にデータが存在する必要があります。
uv run poe das

# 3. 到達可能性(Reachability)の可視化
uv run poe rea
```

### 3. 実験の実行
モデルを指定してQ-Learningによるエッジ除去実験を実行できます。結果は `outputs/` ディレクトリに保存されます。

```bash
# Corticalネットワーク（単調性制約）の実験
uv run poe exp

# Wnt5aネットワーク（休薬制約）の実験
uv run poe exp-wnt5a

# CellCycle10ネットワークの実験
uv run poe exp-cell-cycle10
```

## ディレクトリ構成
- `src/bn_edge_removal/`: コアとなるアルゴリズム（強化学習、ネットワーク定義）
- `scripts/`: 各種実験の実行エントリーポイント
- `configs/`: 実験パラメータ定義。`optimal.yaml` が決定版、`baselines/` に比較用、`paper_reproducibility/` に既報再現用を配置。
- `notebooks/`: Marimo ダッシュボードコード
- `outputs/`: 実験結果出力先（gitignore設定済み）

## 静的Web公開 (WASMデモ) について
Marimo の WASM 機能を利用することで、GitHub Pages上で Python バックエンドなしにダッシュボードをホスティングすることが可能です。
将来的に本リポジトリの Actions を通じて静的サイトとして公開される予定です。

## Citation

本研究の成果（コードまたは手法）を利用する際は、以下の文献を引用してください。

```bibtex
@article{your_citation_key,
  author    = {Author, First and Author, Second},
  title     = {Paper Title describing Boolean Network Edge Removal & Q-Learning},
  journal   = {Journal/Conference Name},
  year      = {2026},
  url       = {https://github.com/koshikai/bn-edge-removal-public}
}
```

## ライセンス
MIT License
