# Boolean Network Specifications

このディレクトリには、ブーリアンネットワークのネットワーク定義ファイル（YAML）が格納されています。

## 概要

| ネットワーク | ノード数 | 削除可能エッジ | 目的状態数 | 分野 | 出典 |
|---|---|---|---|---|---|
| **cell_cycle10** | 10 | 3 | 1 | 細胞周期 (哺乳類) | Faure et al. (2006), Bioinformatics |
| **cortical** | 5 | 6 | 2 | 神経回路 (モノトーン) | 本研究で設計 |
| **s_pombe** | 10 | 4 | 1 | 細胞周期 (分裂酵母) | Davidich & Bornholdt (2008), PLoS ONE（原著準拠） |
| **wnt5a** | 7 | 8 | 2 | シグナル伝達 | 本研究で設計 |
| **yeast_irons2009_n18** | 18 | 0（未選定） | 11 | 細胞周期 (出芽酵母) | Irons (2009)、GINsim。drqn から取り込み |
| **tumor_cohen2015_n20** | 20 | 0（未選定） | 1 | 腫瘍の浸潤・遊走 | Cohen et al. (2015)、GINsim。drqn から取り込み |
| **emt_switch_n12** | 12 | 0（未選定） | 2 | EMT スイッチ | BBM。transfer から取り込み |
| **tlgl_reduced_n18** | 18 | 0（未選定） | 1 | T-LGL 白血病の生存 | BBM。transfer から取り込み |
| **hcc_reduced_n19** | 19 | 0（未選定） | 1 | 肝細胞癌の EMT | BBM。transfer から取り込み |
| **integrated_mammalian_n20** | 20 | 0（未選定） | 2 | 細胞周期（ErbB・ER） | BBM。transfer から取り込み |

---

## 1. cell_cycle10 (哺乳類細胞周期ネットワーク)

### 概要
哺乳類の細胞周期制御ネットワークのブーリアンモデル。G1期停止状態への安定化することを目的とする。

### ノード
| インデックス | ラベル | 説明 |
|---|---|---|
| 1 | Rb | Retinoblastoma tumor suppressor |
| 2 | E2F | Transcription factor |
| 3 | CycE | Cyclin E |
| 4 | CycA | Cyclin A |
| 5 | Cdc20 | Cell division cycle 20 |
| 6 | Cdh1 | Cdc20 homolog 1 |
| 7 | UbcH10 | Ubiquitin-conjugating enzyme |
| 8 | CycB | Cyclin B |
| 9 | p27 | Cyclin-dependent kinase inhibitor |
| 10 | CycD | Cyclin D |

### 目的状態
- `[1, 0, 0, 0, 0, 1, 0, 0, 1, 0]` - G1停止状態 (Rb=1, Cdh1=1, p27=1)

### 削除可能エッジ
| インデックス | ソース | ターゲット | 符号 |
|---|---|---|---|
| 1 | 10 (CycD) | 10 (CycD) | activation |
| 2 | 3 (CycA) | 6 (Cdh1) | inhibition |
| 3 | 5 (Cdc20) | 6 (Cdh1) | inhibition |

### 出典
- Faure, A., Naldi, A., Chaouiya, C., & Thieffry, D. (2006). Dynamical Analysis of a Generic Boolean Model for the Control of the Mammalian Cell Cycle. Bioinformatics, 22(14), e124-e131. https://doi.org/10.1093/bioinformatics/btl210
  - PubMed ID: 16873462
  - GINsimモデルデータベースでも公開: https://ginsim.org/model/boolean-cell-cycle

---

## 2. cortical (皮質神経回路ネットワーク)

### 概要
5ノードのモノトーン・ブーリアンネットワーク。皮質神経回路の inhibition-stabilized 状態への安定化を目的とする。全てのパラメータがinhibition（負）のみで構成される。

### ノード
5つのノード（インデックス1-5）のみで構成される簡略化モデル。

### 目的状態
- `[1, 0, 1, 0, 1]`
- `[0, 1, 0, 1, 0]`

### 削除可能エッジ
6つのエッジ（全てinhibition）。ネットワークの冗長性を削減し、目的状態への到達性を改善する。

### 特徴
- **モノトーン性**: 全てのパラメータがNOT（inhibition）のみ
- **対称性**: 2つの目的状態（交互に活性化するパターン）
- 皮質神経回路の「inhibition stabilization」特性をモデル化

### 出典
このネットワークは、本研究（エッジ除去によるブーリアンネットワーク制御）で設計された5ノードのモノトーン・ネットワークである。

**関連研究**:
- Sanzeni, A., Akitake, B., Goldbach, H. C., Leedy, C. E., Brunel, N., & Histed, M. H. (2020). Inhibition stabilization is a widespread property of cortical networks. eLife, 9, e54875. https://doi.org/10.7554/eLife.54875
- Giacomantonio, C. E., & Goodhill, G. J. (2010). A Boolean Model of the Gene Regulatory Network Underlying Mammalian Cortical Area Development. PLoS Computational Biology, 6(9), e1000936.

---

## 3. s_pombe (分裂酵母細胞周期ネットワーク)

### 概要
分裂酵母 (*Schizosaccharomyces pombe*) の細胞周期制御ネットワーク。G1期停止状態への安定化を目的とする。

### ノード
| インデックス | ラベル | 説明 |
|---|---|---|
| 1 | Start | 外部シグナル (Start)。自己分解し 1 ステップで 0 になる |
| 2 | SK | S-phase promoting kinase |
| 3 | Cdc2_Cdc13 | M-phase promoting kinase (MPK) |
| 4 | Cdc2_Cdc13_star | 活性型MPK |
| 5 | Ste9 | S-phase inhibitor |
| 6 | Rum1 | S-phase inhibitor |
| 7 | Slp1 | M-phase trigger |
| 8 | PP | Phosphatase |
| 9 | Cdc25 | Phosphatase activator |
| 10 | Wee1_Mik1 | Kinase inhibitor |

### 目的状態
- `[0, 0, 0, 0, 1, 1, 0, 0, 0, 1]` - G1 停止状態（Ste9・Rum1・Wee1_Mik1 が ON）

目的の定義（`env.trajectory_satisfies_goal`）は他のネットワークと同じ 2 条件である。

- **到達条件**: ある `k <= reach_horizon` で `x(k)` が目的状態である。
- **維持条件**: `t = k..max_steps` のすべてで `x(t)` が目的状態集合に留まる。

目的状態は u=0 の固定点なので、到達後は何もしなければ維持される（16 通りの入力のうち 12 通りで維持。
u1 と u2 を同時に 1 にすると Cdc2_Cdc13 が ON になり目的状態を離れる）。
`Start` は原著どおり自己分解し、どの状態・どの入力でも 1 ステップで 0 になる（`x1' = 0`）。
介入なし（u=0）で目的状態に入るのは 762/1024 状態、制御を使えば全 1024 状態から到達できる（いずれも 9 ステップ以内）。
そのため `controllable_initial_states_note` は持たない（2026-10-05 までは Start を定数入力として保持しており、
`x1 == 0` の 512 状態に限っていた）。

### 削除可能エッジ
| インデックス | ソース | ターゲット | 符号 |
|---|---|---|---|
| 1 | 5 (Ste9) | 3 (Cdc2_Cdc13) | inhibition |
| 2 | 6 (Rum1) | 3 (Cdc2_Cdc13) | inhibition |
| 3 | 7 (Slp1) | 3 (Cdc2_Cdc13) | inhibition |
| 4 | 10 (Wee1_Mik1) | 4 (Cdc2_Cdc13_star) | inhibition |

制御ビット 1 本はエッジ 1 本に対応し、そのエッジのターゲットの更新則だけを変える。
Ste9 / Rum1 / Slp1 から Cdc2_Cdc13_star への抑制（5→4, 6→4, 7→4）は別の固定エッジである。

### 全エッジ（27 本）
u=0 の更新則は原著（PLoS ONE 3(2): e1672）の閾値ネットワーク（Table 1、Figure 1、閾値は `θ = ±0.5` と読む）と
10 ノードすべて・全 1024 状態で一致する。`all_edges` はその依存関係の全列挙である（Table 1 の 23 本 +
Ste9・Rum1・Cdc25・Wee1_Mik1 の自己依存 4 本）。`Start` の次の値はどのノードにも依存しない（常に 0）ので、
`Start` をターゲットとするエッジは無い。

| ターゲット | 活性化 (activation) | 抑制 (inhibition) |
|---|---|---|
| 1 Start | - | - |
| 2 SK | 1 | - |
| 3 Cdc2_Cdc13 | - | 5 (u1), 6 (u2), 7 (u3) |
| 4 Cdc2_Cdc13_star | 9 | 5, 6, 7, 10 (u4) |
| 5 Ste9 | 5, 8 | 2, 3, 4 |
| 6 Rum1 | 6, 8 | 2, 3, 4 |
| 7 Slp1 | 4 | - |
| 8 PP | 7 | - |
| 9 Cdc25 | 3, 9 | 8 |
| 10 Wee1_Mik1 | 8, 10 | 3 |

> **注記（2026-10-05 照合）**: このネットワークはどの実行スクリプトからも使われていない。
> 2026-10-05 に spec と実装をエッジ単位で全数照合し、出典（biodivine id-095）を正として両方を直した。
> それ以前は (1) `all_edges` が模式的な 19 本で実装と一致せず、(2) 実装の Cdc25 の更新則が出典と異なり、
> (3) 出典に無い PP ⊣ Cdc2_Cdc13_star を削除可能エッジ u4 として持ち、(4) u1〜u3 が 5→4 / 6→4 / 7→4 も
> 同時に打ち消し、(5) 目的状態が固定点でない `[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]` だった。
> 経緯と照合表は `docs/experiments/2026-10-05_implementation_audit.md` を参照。
>
> **注記（2026-10-05 原著準拠）**: その後、原著の Table 1 / Table 2 / Table 3 と直接照合し
> （`docs/experiments/2026-10-05_s_pombe_source_check.md`）、唯一の相違だった `Start` を原著どおり自己分解に
> 変えた（`docs/experiments/2026-10-05_s_pombe_source_conformance.md`）。biodivine id-095 は `Start` を
> 値を保持する入力として扱う点だけが異なる。

### 出典
- Davidich, M. I., & Bornholdt, S. (2008). Boolean Network Model Predicts Cell Cycle Sequence of Fission Yeast. PLoS ONE, 3(2), e1672. https://doi.org/10.1371/journal.pone.0001672
  - PubMed ID: 18301750
  - このモデルは生物学的相互作用のトポロジーのみが知識に基づいて構築され、実験的に知られる細胞周期の活性シーケンスを再現する。
- 更新則は上記原著を正とする。参考: biodivine-boolean-models（https://github.com/sybila/biodivine-boolean-models）の id-095 `FISSION-YEAST-2008`（`model.bnet`、リポジトリ commit `314b2fe` で確認。GINsim のモデルを無改変で収録したもの）は `Start` 以外の 9 ノードの規則が同一で、`Start` だけを入力（更新則なし、値を保持）として扱う。

---

## 4. wnt5a (Wnt5aシグナル伝達ネットワーク)

### 概要
Wnt5aシグナル伝達経路のブーリアンモデル。2つの目的状態を持ち、非回復（non-recovery）と回復（recovery）の状態遷移を制御する。

### ノード
7つのノードで構成されるWnt5a関連シグナル伝達ネットワーク。

### 目的状態
- `[1, 0, 0, 0, 0, 0, 1]` - 非回復状態
- `[0, 1, 0, 1, 1, 1, 1]` - 回復状態

### 削除可能エッジ
| インデックス | ソース | ターゲット | 符号 |
|---|---|---|---|
| 1 | 2 | 2 | activation |
| 2 | 4 | 2 | activation |
| 3 | 6 | 2 | activation |
| 4 | 2 | 5 | activation |
| 5 | 7 | 5 | inhibition |
| 6 | 3 | 6 | activation |
| 7 | 4 | 6 | activation |
| 8 | 2 | 7 | inhibition |

### 特徴
- **2つの目的状態**: 細胞運命の2つの異なる均衡状態
- **制御可能性**: `x2 == 0 or x7 == 0` の初期条件が必要

### 出典
このネットワークは、本研究（エッジ除去によるブーリアンネットワーク制御）で設計された7ノードのWnt5aシグナル伝達モデルである。

**関連研究**:
- Kumawat, K., & Gosens, R. (2016). WNT-5A: signaling and functions in health and disease. Cellular and Molecular Life Sciences, 73(3), 567-587. https://doi.org/10.1007/s00018-015-2076-y
- Siegle, L., Schwab, J. D., Kühlwein, S. D., Lausser, L., Tümpel, S., Pfister, A. S., ... & Kestler, H. A. (2018). A Boolean network of the crosstalk between IGF and Wnt signaling in aging satellite cells. PLOS ONE, 13(3), e0195126.
- Yachie-Kinoshita, A., Onishi, K., Ostblom, J., Posfai, E., Rossant, J., & Zandstra, P. W. (2018). Modeling signaling-dependent pluripotent cell states with Boolean logic can predict cell fate transitions. Molecular Systems Biology, 14(1), e77952.

---

## ファイル構造

各YAMLファイルは以下の構造を持つ：

```yaml
name: <ネットワーク名>
n_nodes: <ノード数>
m_edges: <削除可能エッジ数>
goal_states:
  - [<目的状態1>]
  - [<目的状態2>]
node_labels:  # オプション
  - <ラベル1>
  - <ラベル2>
removable_edges:
  - index: <インデックス>
    source: <ソースノード>
    target: <ターゲットノード>
    sign: <activation|inhibition>
all_edges:
  - source: <ソースノード>
    target: <ターゲットノード>
    sign: <activation|inhibition>
    removable: <true|false>
    removable_index: <インデックス>  # removable=trueの場合
update_equations:
  - "x<num>' = <ブール方程式>"
```

### 制御変数 (u1, u2, ...)
削除可能エッジは対応する制御変数（u1, u2, ...）でモデル化される。エッジが「削除された」状態では対応する制御変数が1となり、エッジの効果が無効化される。

---

## yeast_irons2009_n18 / tumor_cohen2015_n20（drqn から取り込み）

`scripts/import_ginsim_from_drqn.py` が drqn の GINsim モデルから生成する。手で編集しない。除去可能エッジはまだ選んでいない（`m_edges: 0`）。取り込みの確認と制御なしの到達歩数は [2026-11-01 の記録](../../../../docs/experiments/2026-11-01_import_yeast_tumor.md) にある。

## BBM の 4 モデル（transfer から取り込み）

`scripts/import_bbm_from_transfer.py` が transfer の固定コミットの `.bnet` と `bbm_metadata.yaml` から生成する。手で編集しない。ノードは、ファイルの target の順に並べ、右辺にだけ現れる変数（自由な入力）を後ろに加えた。ラベルは先頭の `v_` を除いた。`goal_states` は、制御なしのアトラクタのうち、メタデータの表現型の目標（ZEB1 = 1、Apoptosis_ = 1、ZEB1 = 1、pRB = 1）の中にあるものの状態である。表現型の目標そのものは解析スクリプトで指定する。符号の決まらない依存がある MAPK（`001_mapk_reduced_2`）は取り込んでいない。

