# GeoTRE — Event Temporal Relation Extraction via Geometric Representations

> Undergraduate thesis — A **dual-branch (semantic + geometric)** model that represents every event as a **time interval `[start, end]`** on a continuous axis, then infers temporal relations using **Allen's Interval Algebra**.

---

## 1. Introduction

**Event Temporal Relation Extraction (ETRE)** is the task of identifying the temporal relation between two events in text (e.g., event A happens *before*, *after*, *simultaneously with*, or *contains* event B).

This thesis proposes **GeoTRE**, a model combining two complementary branches:

- **Semantic branch:** classifies the relation directly from the contextual representations of the two events (BERT/RoBERTa + cross-attention).
- **Geometric branch:** maps each event to an interval coordinate `[s, e]` (start, end) on a continuous timeline, then computes **explicit geometric logits for each relation** based on the relative position of the two intervals (in the spirit of Allen's Interval Algebra).

The two branches are constrained to **agree** with each other, while the coordinates of the same event (appearing across multiple pairs) are forced to be **consistent**. At inference time, the two branches are **ensembled** with a weight `α` tuned on the validation set.

---

## 2. Core Idea

### 2.1. Representing events as time intervals

For each event, the model produces a coordinate:

```
s = start                      (start position)
e = s + softplus(Δ) + 0.05     (end is always ≥ start, with a minimum duration)
```

This representation is obtained by taking the vector of the event marker token (`<e1>`, `<e2>`), enriching it via **top-k cross-attention** over the full context, then passing it through a `geo_head`.

### 2.2. Geometric logits from Allen's Interval Algebra

Given two intervals `E1 = [s1, e1]` and `E2 = [s2, e2]`, the logits are computed explicitly, e.g.:

| Relation | Logit formula (higher ⇒ more likely) |
|---|---|
| **BEFORE** | `s2 - e1` (E1 ends before E2 starts) |
| **AFTER** | `s1 - e2` (E2 ends before E1 starts) |
| **INCLUDES** | `min(s2 - s1, e1 - e2) - tol` (E1 fully contains E2) |
| **IS_INCLUDED** | `min(s1 - s2, e2 - e1) - tol` (E2 fully contains E1) |
| **SIMULTANEOUS / EQUAL** | `2·tol - (|s2 - s1| + |e1 - e2|)` (the two intervals nearly coincide) |

Here both `tol` (tolerance threshold) and `temperature` are **learnable parameters**. The final logits are divided by the temperature before softmax.

### 2.3. Multi-component loss function

```
total = CE(semantic)                          # semantic classification
      + λ_gce   · CE(geometric)               # geometric (Allen-based) classification
      + λ_con   · consistency_loss            # same event ⇒ same coordinates
      + λ_align · symmetric_KL(semantic, geo) # the two branches agree
      + λ_vague · vague_separation_loss        # separate VAGUE / non-VAGUE (MATRES, TBD only)
```

- **Consistency loss:** for an event appearing in multiple pairs (grouped per document by `SmartBatchSampler`), force its `(s, e)` coordinates to converge to a single value.
- **Alignment loss:** symmetric KL between the two branches' predicted distributions, enforcing consistency.
- **Vague separation:** a dedicated binary branch to detect ambiguous (VAGUE) relations.
- **Class weights:** handle class imbalance.

### 2.4. Inference

On the validation set, the model **searches for the optimal weight `α`** to combine:

```
prob = α · softmax(logits_semantic) + (1 − α) · softmax(logits_geometric)
```

Test results are reported for all three strategies: **Semantic**, **Geometric**, and **Ensemble**.

---

## 3. Datasets

All data has been preprocessed into a single CSV format, stored as `.zip` files in [`Data/`](Data/):

| File | Dataset | #Classes | Splits | Labels |
|---|---|---|---|---|
| `MATRES.zip` | **MATRES** | 4 | train / test | BEFORE, AFTER, EQUAL, VAGUE |
| `TBD.zip` | **TB-Dense** | 6 | train / dev / test | BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS, VAGUE |
| `TDDAutoProcessed.zip` | **TDDiscourse (Auto)** | 5 | train / dev / test | b, a, i, ii, s |
| `TDDManProcessed.zip` | **TDDiscourse (Man)** | 5 | train / dev / test | b, a, i, ii, s |

**Structure of each CSV row:**

```
entity1_id, entity2_id,
entity1_start, entity2_start, entity1_end, entity2_end,   # character offsets of the two events in the text
entity1_text, entity2_text,
document_id,                                               # used for consistency loss & batch sampler
text,                                                      # original text passage
label                                                      # relation label
```

During preprocessing, the model inserts `<e1></e1>`, `<e2></e2>` tags around the two events and applies smart context windowing around them (up to 256 tokens).

> ⚠️ The data paths inside the notebooks follow the **Kaggle** environment (`/kaggle/input/...`). When running elsewhere, unzip the archives above and update the `pd.read_csv(...)` paths accordingly.

---

## 4. Repository Structure

```
.
├── Code/
│   ├── Matres/                     # Experiments on MATRES (4 classes)
│   │   ├── bert-base/              #   - BERT, 3 seeds (42, 123, 567)
│   │   ├── roberta-base/           #   - RoBERTa, 3 seeds
│   │   └── Các cấu hình loại bỏ/   #   - Ablations: no_align / no_con / no_geo_branch
│   │
│   ├── TD-Dense/                   # Experiments on TB-Dense (6 classes)
│   │   ├── bert/
│   │   ├── roberta/
│   │   └── Các cấu hình loại bỏ/
│   │
│   ├── TDD-Auto/                   # Experiments on TDDiscourse-Auto (5 classes)
│   │   ├── bert-base/
│   │   ├── roberta-base/
│   │   └── Các thiết lập loại bỏ/
│   │
│   ├── TDD-Man/                    # Experiments on TDDiscourse-Man (5 classes)
│   │   ├── bert-base/
│   │   ├── roberta-base/
│   │   └── Các thiết lập loại bỏ/
│   │
│   └── Phân tích/                  # In-depth analysis notebooks
│       ├── Phân tích tọa độ.ipynb              # Visualize the learned [s, e] coordinates
│       └── Phân tích dự đoán Vague ...ipynb    # Error analysis on the VAGUE class
│
└── Data/                           # Preprocessed data (.zip)
    ├── MATRES.zip
    ├── TBD.zip
    ├── TDDAutoProcessed.zip
    └── TDDManProcessed.zip
```

> Note: some folders use Vietnamese names — *"Các cấu hình loại bỏ" / "Các thiết lập loại bỏ"* mean **"ablation configurations"**, and *"Phân tích"* means **"Analysis"**.

### Notebook naming convention

```
geo-<dataset>_<config>_<encoder>_seed_<seed>.ipynb

e.g.: geo-matres_full_bert_seed_123.ipynb        → MATRES, full model, BERT, seed 123
      geo-tbd_no_align_loss_bert_seed42.ipynb    → TB-Dense, no alignment loss, BERT, seed 42
```

**Ablation configurations** (used to demonstrate the contribution of each component):

| Suffix | Meaning |
|---|---|
| `full` | Full model |
| `no_geo_branch` | Remove the geometric branch (semantic branch only) |
| `no_align(ment)_loss` | Remove the agreement constraint between the two branches |
| `no_con_loss` | Remove the coordinate consistency constraint |
| `no_con_alignment` | Remove both constraints above at once |

---

## 5. Model Architecture (`GeoTREModel`)

<img width="1123" height="511" alt="1779276783258_7888937454975708662_g7789413039228685972_e76da13ff536e24165b8358d6a871318" src="https://github.com/user-attachments/assets/621d39bd-39d6-4219-8027-4f26480fec4d" />

**Key components:**

- **Encoder** — BERT-base-uncased or RoBERTa-base; encodes the input text (with `<e1>`/`<e2>` markers) and yields the two event representations **h₁, h₂** (enriched via top-k cross-attention over the context).
- **Geometric Branch** — each representation passes through an **FFN** (`geo_head`) that predicts an interval coordinate **(sᵢ, eᵢ)** on the timeline. The **Allen Interval Decoder** then turns the two intervals into relation logits using explicit interval-algebra formulas, followed by a Softmax.
- **Semantic Branch** — the two representations are concatenated (**⊕**, `[h₁ ; h₂]`), passed through an **FFN** (`cls_head`), and a Softmax to predict the relation directly from context.
- **Late Fusion** — the two branches' probability distributions are combined with a learned weight **α** (tuned on the validation set) to produce the final relation (BEFORE / AFTER / overlap relations …).
- **Auxiliary** — a `vague_head` (binary VAGUE vs. non-VAGUE separation) and the learnable parameters `simul_threshold` and `geo_temp` inside the Allen decoder (tolerance / temperature).


---

## 6. Main Hyperparameters

| Parameter | Value |
|---|---|
| Encoder | `bert-base-uncased` / `roberta-base` |
| Max length | 256 |
| Batch size | 32 |
| Optimizer | AdamW (`lr = 2e-5`, `weight_decay = 0.01`) |
| Scheduler | Linear warmup (10% of steps) |
| Epochs | 20–40 (early stopping, patience = 5) |
| `λ_con` / `λ_gce` / `λ_align` / `λ_vague` | 0.1 / 0.5 / 0.1 / 0.1 |
| Seeds | 42, 123, 567 (and 678 in some configs) |
| Evaluation | Micro-F1 & Macro-F1 (VAGUE excluded from F1) |

---

## 7. How to Run

Each notebook is a self-contained, end-to-end experiment (load data → preprocess → train → evaluate → confusion matrix).

1. **Environment:** **Kaggle / Google Colab with GPU** is recommended (notebooks use Kaggle paths).
2. **Libraries:** `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`.
3. **Data:** download the data from [`Data/`](Data/), unzip it, and point `pd.read_csv(...)` to the correct `train.csv` / `dev.csv` / `test.csv` paths.
4. **Run:** open the relevant notebook (dataset / encoder / seed) and run the cells top to bottom.
5. **Results:** the notebook prints a classification report + Micro/Macro-F1 for all three strategies (Semantic / Geometric / Ensemble) along with confusion matrices.

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

---

## 8. Analysis Notebooks

- **`Phân tích tọa độ.ipynb` (Coordinate analysis)** — Visualizes the learned interval coordinates `[s, e]`, verifying that the geometric relations (before/after/inclusion) are indeed correctly reflected on the timeline.
- **`Phân tích dự đoán Vague trên TB-Dense và Matres.ipynb` (VAGUE prediction analysis)** — Error analysis focused on the **VAGUE** class (the hardest and noisiest class), assessing the effectiveness of the VAGUE separation branch.

---

## 9. Summary of Contributions

1. **Interpretable geometric representation:** each event = an interval `[s, e]`; temporal relations are derived via explicit Allen-style formulas instead of a pure black box.
2. **Two-branch architecture + agreement constraint:** combines the strength of semantic classification with geometric consistency.
3. **Event-level coordinate consistency:** the same event must hold the same temporal position even when it appears in multiple pairs.
4. **Comprehensive evaluation:** 4 datasets (MATRES, TB-Dense, TDD-Auto, TDD-Man) × 2 encoders (BERT, RoBERTa) × multiple seeds, with full ablation studies.

---

*Undergraduate thesis — 2025.*
