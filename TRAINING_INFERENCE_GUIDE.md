# Transaction Tagger — Training & Inference Guide

## Architecture at a Glance

```
Training CSV (35K labeled, 43 categories)
        │
        ▼
  FusionEncoder  (BERT + categorical embeddings + numeric → 256-D L2 unit vector)
        │
        ▼  SupCon loss + class-balanced batches + inv-freq weighting
        │
  fusion_encoder_best.pth
        │
        ▼
  GoldenRecordIndexer  ──► Stratified FAISS HNSW index
                       ──► Per-class prototypes [43 × 256]
                       ──► OOD distance threshold (calibrated from rgs.csv)
        │
        ▼
  TransactionInferencePipeline.predict_batch()
    ├── GPU BERT encoding (FP16, batched)
    ├── FAISS k-NN retrieval → distance-weighted vote
    ├── Batched GPU prototype scoring (one matmul for all N queries)
    ├── Combined score → predicted_category + confidence
    ├── OOD flag  (min_dist > threshold)
    └── LLM fallback  (optional, for needs_llm=True transactions)
```

---

## 0. Install Dependencies

### Full package list (by source file)

| Package | pip name | Used by | Required? |
|---|---|---|---|
| PyTorch | `torch` | All model files | ✅ Core |
| HuggingFace Transformers | `transformers` | `fusion_encoder`, `data_loader`, `inference_pipeline` | ✅ Core |
| FAISS (CPU) | `faiss-cpu` | `inference_pipeline` | ✅ Core |
| FAISS (GPU) | `faiss-gpu` | `inference_pipeline` (`--gpu-index`) | ⚡ GPU only |
| NumPy | `numpy` | All numerical files | ✅ Core |
| Pandas | `pandas` | `data_loader`, `inference_pipeline`, `train_multi_expt` | ✅ Core |
| scikit-learn | `scikit-learn` | `data_loader` (StandardScaler), `validation` (F1, AUROC), `plotting` (t-SNE) | ✅ Core |
| tqdm | `tqdm` | `train_multi_expt` (progress bar) | ✅ Core |
| PyYAML | `pyyaml` | `train_multi_expt` (config loading) | ✅ Core |
| Matplotlib | `matplotlib` | `plotting` (loss curves, grad norms) | ✅ Core |
| Requests | `requests` | `llm_classifier` (Ollama HTTP calls) | 🔌 LLM optional |
| Ollama | (desktop app) | `llm_classifier` — local llama3.2 inference | 🔌 LLM optional |

### Install commands

#### CPU only (no GPU)
```bash
pip install torch transformers faiss-cpu scikit-learn pandas numpy tqdm pyyaml matplotlib requests
```

#### GPU — CUDA 11.8
```bash
# PyTorch with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# FAISS GPU (IVF/Flat index on GPU — HNSW stays on CPU regardless)
pip install faiss-gpu

# Everything else
pip install transformers scikit-learn pandas numpy tqdm pyyaml matplotlib requests
```

#### GPU — CUDA 12.1
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
pip install transformers scikit-learn pandas numpy tqdm pyyaml matplotlib requests
```

#### GPU — CUDA 12.4+ / PyTorch 2.4+
```bash
pip install torch
pip install faiss-gpu-cu12          # CUDA 12.x build of faiss-gpu
pip install transformers scikit-learn pandas numpy tqdm pyyaml matplotlib requests
```

#### LLM fallback — Ollama (local, no API key needed)
```bash
# 1. Install Ollama desktop app: https://ollama.com/download
# 2. Pull the model
ollama pull llama3.2

# 3. Start the server (keep this running in a terminal)
ollama serve

# 4. Install the requests package (used by llm_classifier.py to call Ollama)
pip install requests
```

### requirements.txt

```
# Core — required for training and inference
torch>=2.0.0
transformers>=4.35.0
faiss-cpu>=1.7.4        # replace with faiss-gpu for GPU FAISS index support
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pyyaml>=6.0
matplotlib>=3.7.0

# LLM fallback — required only when using --use-llm
requests>=2.28.0        # HTTP calls to local Ollama server
```

> **Note on faiss-gpu:** `faiss-gpu` only accelerates **IVF and Flat** index types.
> The default **HNSW** index cannot run on GPU — it always uses CPU search.
> BERT encoding runs on GPU regardless of which FAISS variant you install.

---

## 1. Save Training Artifacts

Build vocabulary, scaler, and label mapping from your labeled CSV.
Run **once** before training; re-run only if your training data changes.

```bash
python save_training_artifacts.py \
  --csv         data/golden_records.csv \
  --ood-label   other \
  --bert-model  bert-base-uncased \
  --output-dir  training_artifacts
```

**Output:**
```
training_artifacts/
  training_artifacts.pkl   ← cat_vocab, scaler, label_mapping, class_counts
  model_config.json        ← human-readable config
```

**Verify it worked:**
```bash
python - <<'EOF'
import pickle
with open("training_artifacts/training_artifacts.pkl", "rb") as f:
    a = pickle.load(f)
print(f"Categories : {len(a['label_mapping'])}")          # should be 43
print(f"Vocab sizes: {a['categorical_dims']}")
sorted_counts = sorted(a['class_counts'].items(), key=lambda x: x[1])
print(f"Min class  : {sorted_counts[0]}")                  # ~250 for minority
print(f"Max class  : {sorted_counts[-1]}")                 # ~3000-5000 for majority
EOF
```

---

## 2. Training

### Step 2a — Smoke Test (2-3 min, CPU)

Always run this first to verify the pipeline doesn't crash before a full training run.

```bash
python -m src.train_multi_expt --single '{
  "csv_path":           "data/golden_records.csv",
  "categorical_cols":   ["tran_mode", "dr_cr_indctor", "sal_flag"],
  "numeric_cols":       ["tran_amt_in_ac"],
  "label_col":          "category",
  "ood_label":          "other",
  "bert_model":         "bert-base-uncased",
  "text_proj_dim":      256,
  "final_dim":          256,
  "freeze_strategy":    "freeze",
  "loss_type":          "supcon",
  "temperature":        0.07,
  "balanced_sampling":  true,
  "inv_freq_weighting": true,
  "epochs":             3,
  "batch_size":         64,
  "lr":                 0.00002,
  "val_split":          0.15,
  "patience":           2,
  "sample_size":        3000
}'
```

**Expected output (confirms pipeline is working):**
```
Excluded 'other' OOD label: 3500 → 3000 records
Train samples: 2550, Validation samples: 450
Classes: 43 | min weight: 1.00 | max weight: 4.73
Running experiment: tagger_proj256_final256_freeze-freeze_bs64_lr2.00e-05_supcon on cpu

Epoch [1/3], Batch [3], Loss: 3.8421
Epoch [1/3] Train Loss: 3.7241
Evaluating on validation set...
Validation Report - Epoch 1
  Accuracy  : 0.2341
  Recall@5  : 0.5812
  Macro-F1  : 0.1923
Validation metric improved to 0.5812
New best model! Saving to experiments/.../fusion_encoder_best.pth
```

### Step 2b — Full Training Run (recommended config)

Experiment 2 from `experiments_with_validation.yaml` — SupCon + balanced sampling + gradual BERT unfreezing. Best choice for imbalanced 43-class real data.

```bash
python -m src.train_multi_expt --single '{
  "csv_path":           "data/golden_records.csv",
  "categorical_cols":   ["tran_mode", "dr_cr_indctor", "sal_flag"],
  "numeric_cols":       ["tran_amt_in_ac"],
  "label_col":          "category",
  "ood_label":          "other",
  "bert_model":         "bert-base-uncased",
  "text_proj_dim":      256,
  "final_dim":          256,
  "freeze_strategy":    "gradual",
  "loss_type":          "supcon",
  "temperature":        0.07,
  "balanced_sampling":  true,
  "inv_freq_weighting": true,
  "semi_hard_mining":   true,
  "epochs":             30,
  "batch_size":         128,
  "lr":                 0.00002,
  "val_split":          0.15,
  "patience":           7,
  "min_delta":          0.001
}'
```

Or run all 4 configured experiments in sequence:
```bash
python -m src.train_multi_expt --config experiments_with_validation.yaml
```

### What to Monitor During Training

| Console output | What it means |
|---|---|
| `Excluded 'other' OOD label: N → M records` | OOD rows correctly removed before training |
| `Classes: 43 \| min weight: 1.00 \| max weight: ~4.5` | Inv-freq weighting is active |
| `Unfreezing top N` (N increases each epoch) | Top-down BERT unfreezing working (layer 11→10→9...) |
| Train Loss decreasing | Embedding space is separating categories |
| `Recall@5` improving | Model retrieves same-class neighbors more often |
| `Macro-F1` improving | Minority classes improving, not just majority |
| `New best model! Saving ...` | Checkpoint saved on best Recall@5 |
| `Early stopping triggered at epoch N` | Training converged; best checkpoint is at best epoch |

### Training Output

```
experiments/
  tagger_proj256_final256_freeze-gradual_bs128_lr2.00e-05_supcon/
    fusion_encoder_best.pth        ← use this for inference
    fusion_encoder_epoch_3.pth     ← periodic checkpoint
    logs/
      training_logs.json           ← all epoch losses + validation metrics
    plots/
      loss_curve.png
      grad_norms.png
      validation_metrics.png
```

---

## 3. Build the FAISS Index

Run once per trained checkpoint. Encodes the full golden record corpus and builds the retrieval index.

```bash
python run_inference.py \
  --artifacts   training_artifacts/training_artifacts.pkl \
  --model       experiments/tagger_proj256_final256_freeze-gradual_bs128_lr2.00e-05_supcon/fusion_encoder_best.pth \
  --csv         data/golden_records.csv \
  --ood-csv     data/rgs.csv \
  --index       golden_records.faiss \
  --index-type  HNSW \
  --max-per-class 500 \
  --batch-size  512
```

> Use `--index-type IVF` instead of `HNSW` if you want GPU FAISS search (add `--gpu-index`).
> HNSW is faster on CPU and is the recommended default.

**What this builds:**
- Encodes all 35K rows in batches of 512 on GPU
- Caps majority classes at 500 diverse representatives (prevents density bias in k-NN)
- Builds FAISS HNSW index (~21,500 stratified vectors)
- Computes L2-normalised prototype per class
- Calibrates OOD threshold from `rgs.csv` "other" rows

**Output:**
```
golden_records.faiss              ← FAISS retrieval index
golden_records_metadata.pkl       ← labels, embeddings, prototypes, ood_threshold
```

---

## 4. Inference

### Option A — CLI (CSV in, CSV out)

```bash
python run_inference.py \
  --skip-build \
  --artifacts  training_artifacts/training_artifacts.pkl \
  --model      experiments/.../fusion_encoder_best.pth \
  --index      golden_records.faiss \
  --input-csv  data/new_transactions.csv \
  --output-csv results.csv \
  --top-k      10 \
  --batch-size 512 \
  --confidence-threshold 0.75
```

#### With LLM fallback for uncertain / OOD transactions:
```bash
python run_inference.py \
  --skip-build \
  --input-csv  data/new_transactions.csv \
  --output-csv results.csv \
  --use-llm \
  --llm-model  llama3.2
```

#### With GPU FAISS search (IVF index only):
```bash
python run_inference.py \
  --skip-build \
  --input-csv   data/new_transactions.csv \
  --output-csv  results.csv \
  --index-type  IVF \
  --gpu-index             # moves FAISS index to GPU — requires faiss-gpu
  --batch-size  1024
```

### Option B — Python API (production, 10M+ scale)

```python
from src.inference_pipeline import TransactionInferencePipeline

# ─── 1. Load once at service startup ────────────────────────────────────────
pipeline = TransactionInferencePipeline(
    artifacts_path="training_artifacts/training_artifacts.pkl",
    model_path="experiments/.../fusion_encoder_best.pth",
    index_path="golden_records.faiss",
    use_fp16=True,           # FP16 on GPU: ~2× throughput, half VRAM
    use_gpu_index=False,     # True only for IVF/Flat index types; HNSW stays on CPU
    hnsw_ef_search=64,       # 64 ≈ 97% recall and fast; 128 ≈ 99% and slightly slower
    confidence_threshold=0.75,
)

# ─── 2. Prepare transactions ─────────────────────────────────────────────────
transactions = [
    {
        "tran_partclr":   "SWIGGY ORDER 23456789",
        "tran_mode":      "UPI",
        "dr_cr_indctor":  "D",
        "sal_flag":       "N",
        "tran_amt_in_ac": 450.0,
    },
    # ... as many as needed
]

# ─── 3. Batch predict ────────────────────────────────────────────────────────
# All encoding happens on GPU in mini-batches.
# Prototype scoring uses a single batched GPU matmul for all N queries.
results = pipeline.predict_batch(
    transactions,
    top_k=10,
    batch_size=512,   # tune to your GPU VRAM (see table below)
)

# ─── 4. Read results ─────────────────────────────────────────────────────────
for txn, r in zip(transactions, results):
    print(
        r["predicted_category"],   # e.g. "Food_restaurants"
        f"{r['confidence']:.0%}",  # e.g. "87%"
        "OOD" if r["is_ood"] else "",
        "→LLM" if r["needs_llm"] else "",
    )
```

### Option C — Full 10M Scale Example

```python
import pandas as pd
from src.inference_pipeline import TransactionInferencePipeline

# Boot once
pipeline = TransactionInferencePipeline(
    artifacts_path="training_artifacts/training_artifacts.pkl",
    model_path="experiments/.../fusion_encoder_best.pth",
    index_path="golden_records.faiss",
    use_fp16=True,
    hnsw_ef_search=64,      # favour speed at this scale
    confidence_threshold=0.75,
)

# Load all transactions
df = pd.read_csv("data/all_transactions.csv")
df = df.rename(columns={"tran_type": "tran_mode"})   # handle column variants
df["tran_partclr"] = df["tran_partclr"].fillna("")
df["tran_mode"]    = df["tran_mode"].fillna("NULL")
df["sal_flag"]     = df["sal_flag"].fillna("NULL")

transactions = df[
    ["tran_partclr", "tran_mode", "dr_cr_indctor", "sal_flag", "tran_amt_in_ac"]
].to_dict("records")

# One call — GPU-batched encoding + FAISS search + batched prototype scoring
results = pipeline.predict_batch(transactions, top_k=10, batch_size=512)

# Write results
df["predicted_category"] = [r["predicted_category"] for r in results]
df["confidence"]         = [r["confidence"]         for r in results]
df["is_ood"]             = [r["is_ood"]             for r in results]
df["needs_llm"]          = [r["needs_llm"]          for r in results]

# Split fast-path vs LLM queue
auto_tagged = df[~df["needs_llm"]]       # ~60-70%: high-confidence predictions
llm_queue   = df[df["needs_llm"]]        # ~30-40%: uncertain or OOD → send to LLM

print(f"Auto-tagged : {len(auto_tagged):,}")
print(f"LLM queue   : {len(llm_queue):,}")

auto_tagged.to_csv("results_auto.csv",      index=False)
llm_queue.to_csv(  "results_llm_queue.csv", index=False)
```

---

## 5. Output Schema

Each prediction in `results` is a dict:

| Field | Type | Description |
|---|---|---|
| `predicted_category` | str | One of 43 category names, or `"other"` if OOD |
| `confidence` | float | Combined k-NN + prototype score in [0, 1] |
| `is_ood` | bool | True if min k-NN distance exceeds calibrated threshold |
| `ood_distance` | float | Min distance to any indexed vector |
| `needs_llm` | bool | True if `is_ood` or `confidence < threshold` |
| `vote_distribution` | dict | `{category: score}` — full breakdown of k-NN + prototype votes |
| `distances` | list[float] | Raw distances to top-k neighbors |
| `top_k_labels` | list[str] | Category labels of top-k retrieved neighbors |

The output CSV from `run_inference.py` additionally includes:
`llm_overridden`, `top3_retrieved_desc`, `top3_retrieved_cats`

---

## 6. Performance Tuning

### batch_size for encoding (BERT on GPU)

| Hardware | Recommended `batch_size` |
|---|---|
| CPU only | 32–64 |
| GPU 8 GB (T4, RTX 3060) | 256–512 |
| GPU 16–24 GB (A10, RTX 3090) | 512–1024 |
| GPU 80 GB (A100) | 2048+ |

### HNSW `efSearch` — accuracy vs latency

| `efSearch` | Recall@10 | Relative latency |
|---|---|---|
| 32 | ~94% | 1× (fastest) |
| 64 | ~97% | 1.5× |
| 128 | ~99% | 2.5× (default) |
| 256 | ~99.9% | 5× |

Use 64 for bulk batch jobs, 128 for interactive / production API.

### FAISS GPU index (IVF index type only)

HNSW cannot run on GPU. To use GPU FAISS search, rebuild the index with `--index-type IVF`:

```bash
# Build IVF index (GPU-compatible)
python run_inference.py \
  --csv data/golden_records.csv --ood-csv data/rgs.csv \
  --index golden_records_ivf.faiss --index-type IVF

# Run inference with GPU FAISS search
python run_inference.py \
  --skip-build \
  --index      golden_records_ivf.faiss \
  --input-csv  data/new_transactions.csv \
  --gpu-index                              # moves IVF index to GPU
  --batch-size 1024
```

Or in Python:
```python
import faiss
# After creating the pipeline, manually move the index to GPU
res = faiss.StandardGpuResources()
pipeline.index    = faiss.index_cpu_to_gpu(res, 0, pipeline.index)
pipeline._gpu_res = res   # keep reference to prevent GC
```

### Approximate throughput

| Setup | batch_size | Throughput |
|---|---|---|
| CPU only | 64 | ~200–500 txn/sec |
| GPU T4, FP32 | 256 | ~2,000–3,000 txn/sec |
| GPU T4, FP16 | 512 | ~4,000–6,000 txn/sec |
| GPU A10, FP16 | 1024 | ~10,000–15,000 txn/sec |
| GPU A100, FP16 | 2048 | ~25,000–40,000 txn/sec |

At 10,000 txn/sec (T4, FP16), 10M transactions completes in ~17 minutes.

---

## 7. Code Review — What Was Fixed

### Bug: Per-sample prototype scoring in `predict_batch`

**Before (slow):** `predict_batch` called `_prototype_scores(query_emb)` inside a Python
`for` loop, performing N separate `[C,D]@[D,1]` numpy matmuls.

**After (fast):** Added `_prototype_scores_batch(all_embeddings)` which computes
`[N,D]@[D,C]` as one GPU tensor matmul before the loop. For N=10,000 this is
~10,000× fewer round-trips to the GPU.

```python
# Before — inside for loop:
proto_scores = self._prototype_scores(query_emb)          # called N times

# After — once before the loop:
all_proto_sims = self._prototype_scores_batch(all_embeddings)   # [N, C], one matmul
# then inside loop just index the row:
proto_scores = {self._proto_labels[j]: float(all_proto_sims[i, j]) for j in ...}
```

### Feature: FAISS GPU index support

Added `use_gpu_index=True` parameter to `TransactionInferencePipeline`. Gracefully
handles the HNSW limitation (HNSW cannot run on GPU) with a clear warning and
automatic CPU fallback:

```python
pipeline = TransactionInferencePipeline(
    ...,
    use_gpu_index=True,   # works with IVF/Flat; warns + falls back for HNSW
)
```

### Feature: Prototype matrix cached on GPU

The `[C, D]` prototype matrix is now stored as a half-precision GPU torch tensor
(`_proto_tensor`) at load time, so `_prototype_scores_batch` never copies data
from CPU to GPU during inference.
