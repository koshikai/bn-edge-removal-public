# Boolean Network Specifications

このディレクトリには、ブーリアンネットワークのネットワーク定義ファイル（YAML）が格納されています。

## 概要

| ネットワーク | ノード数 | 削除可能エッジ | 目的状態数 | 分野 | 出典 |
|---|---|---|---|---|---|
| **cell_cycle10** | 10 | 3 | 1 | 細胞周期 (哺乳類) | Faure et al. (2006), Bioinformatics |
| **cortical** | 5 | 6 | 2 | 神経回路 (モノトーン) | 本研究で設計 |
| **s_pombe** | 10 | 8 | 1 | 細胞周期 (分裂酵母) | Davidich & Bornholdt (2008), PLoS ONE |
| **wnt5a** | 7 | 8 | 2 | シグナル伝達 | 本研究で設計 |

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
| 1 | Start | 外部シグナル (Start) |
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
- `[0, 0, 1, 0, 0, 1, 0, 0, 1, 0]` - G1停止状態

### 削除可能エッジ
| インデックス | ソース | ターゲット | 符号 |
|---|---|---|---|
| 1 | 5 (Ste9) | 3 | inhibition |
| 2 | 6 (Rum1) | 3 | inhibition |
| 3 | 7 (Slp1) | 3 | inhibition |
| 4 | 8 (PP) | 4 | inhibition |
| 5 | 10 (Wee1_Mik1) | 4 | inhibition |
| 6 | 4 | 5 (Ste9) | inhibition |
| 7 | 4 | 6 (Rum1) | inhibition |
| 8 | 4 | 10 (Wee1_Mik1) | inhibition |

### 出典
- Davidich, M. I., & Bornholdt, S. (2008). Boolean Network Model Predicts Cell Cycle Sequence of Fission Yeast. PLoS ONE, 3(2), e1672. https://doi.org/10.1371/journal.pone.0001672
  - PubMed ID: 18301750
  - このモデルは生物学的相互作用のトポロジーのみが知識に基づいて構築され、実験的に知られる細胞周期の活性シーケンスを再現する。

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
