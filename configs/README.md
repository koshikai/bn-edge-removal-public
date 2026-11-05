# configs/

実験パラメータ（YAML）。1 つの設定につき 1 ファイルで、seed は含めて実行時に `--seed NN` で上書きする（`run_cortical.py`、`run_wnt5a.py`）。

```bash
uv run python scripts/run_cortical.py --config configs/cortical_monotone.yaml --seed 12
```

## 置き場

| 場所 | 内容 |
|---|---|
| 直下 | 現役の基本設定。`poe` タスクが参照する（cortical / wnt5a / cell_cycle10 / jcmsi_hard） |
| `papers/<paper>/` | 論文の図・数値に使う設定。`output_dir: outputs/papers/<paper>`、`record_summary: false`。対応表は [docs/papers/](../docs/papers/README.md) |
| `replication_2026-10-08/` | 2026-10-08 の SCI'26 本文設定の再実行で使った設定（cortical の 1 要因切り分け 14 腕、本文設定 3 系統）。記録は [docs/experiments/2026-10-08_sci26_text_replication.md](../docs/experiments/2026-10-08_sci26_text_replication.md) |
| `legacy/` | 過去の記録が名前で引く設定（tuned 系、masked ベースライン、ep24000/36000）。新しい実験では使わない |

## 旧パスの読み替え

2026-10-20 より前は、seed ごとに別ファイルだった（`<name>_seed11.yaml` 〜 `_seed15.yaml`、seed 行以外は同一）。
1 つにまとめたので、記録や manifest にある旧パスは次のように読み替える。

| 記録にあるパス | 現在 |
|---|---|
| `configs/<name>_seed<NN>.yaml`（`cortical_split_*`、`cortical_monotone_papertext`、`wnt5a_recovery_paper_text*`） | `configs/replication_2026-10-08/<name>.yaml` を `--seed <NN>` で実行 |
| `configs/papers/<paper>/<model>_seed<NN>.yaml` | `configs/papers/<paper>/<model>.yaml` を `--seed <NN>` で実行 |
| `configs/cortical_monotone_tuned*.yaml`、`cortical_qonly_baseline_masked.yaml`、`wnt5a_recovery_ep*.yaml` | `configs/legacy/` の同名ファイル |

集まった run の `params.json` / `metrics.json` には、実行時の config 全体が保存されている（seed と `output_dir` を含む）。
`scripts/run_config_split.py` は manifest の旧パスを上の規則で解決する。
