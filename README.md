# MYST-ReID

### Official implementation of the paper:
#### - MYST: Benchmarking Ecological and Cross-Medium Generalization in Sea Turtle Re-Identification

ICIP2026 | [Paper](https://ieeexplore.ieee.org/abstract/document/11630082) | [Bibtex](#citation)

## Introduction:
Automated animal re-identification (Re-ID) is vital for longitudinal ecological monitoring. However, current benchmarks largely overlook two critical generalization challenges: performance degradation across unseen ecological contexts and drastic optical shifts. To bridge this gap, we introduce **MYST (MalaYsian Sea Turtle)**, a novel benchmark designed to evaluate model robustness to the **ecological gap** (captivity-to-wild) and the **medium gap** (aquatic-to-terrestrial). We further propose a **Chronological Open-Set (COS) Protocol** that utilizes the DIR@FPIR metric to couple identification accuracy with impostor rejection.

Benchmarking state-of-the-art models reveals critical generalization failure modes. While the MegaDescriptor excels on its source domain (96% Rank-1), it fails to generalize across the ecological gap, underperforming the hand-crafted SIFT descriptor (70% vs. 76%), despite being trained on the same species in rehabilitation settings. More critically, under the medium shift of the nesting environment, MegaDescriptor collapses to 22% DIR@1%FPIR, indicating a high risk of identity merging. In contrast, the learned local descriptor ALIKED demonstrates superior robustness, achieving not only high 97% Rank-1 accuracy but also robust open-set reliability (91.7% DIR@1%FPIR) by exploiting the pose stability of nesting data.

*Codebase Reference: [wildlife-tools](https://github.com/WildlifeDatasets/wildlife-tools) and [wildlife-datasets](https://github.com/WildlifeDatasets/wildlife-datasets).*

## Installation

### Requirements

- Python >= 3.11
- PyTorch (with CUDA for GPU inference)
- hydra-core 1.3.2
- wildlife-datasets 1.0.7
- wildlife-tools 1.0.3 (included as a pinned submodule)
- loguru 0.7.3

### Install MYST-ReID

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. `wildlife-tools` is included as a
pinned git submodule and resolved as a uv workspace member, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/nwjun/myst-reid.git
cd myst-reid
uv sync
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Dataset

| Dataset | Role | Source |
|---|---|---|
| **MYST** (Mantanani, Sipadan, Pom Pom, Redang) | Target domain (ours) | *To be released* — see [Dataset Availability](#dataset-availability) |
| [SeaTurtleIDHeads](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads) | Source domain baseline | Downloaded automatically via `wildlife-datasets` |

Place all datasets in the `./data` folder. If you prefer a different location, override `dataset.root`
(see [Configuration Override](#configuration-override)).

The default configuration expects the following folder structure:

```bash
./data
    ├── mantanani
    │   ├── images/
    │   └── metadata.csv
    ├── sipadan
    │   ├── images/
    │   └── metadata.csv
    ├── pompom_j
    │   ├── images/
    │   └── metadata.csv
    ├── pompom_s
    │   ├── images/
    │   └── metadata.csv
    ├── redang                 # MYST-Terrestrial benchmark
    │   ├── images/
    │   └── metadata.csv
    ├── merged_aquatic         # MYST-Aquatic benchmark (Mantanani + Sipadan + Pom Pom)
    │   └── metadata.csv       # NOTE: `path` column is relative to ./data, not to this folder
    └── SeaTurtleIDHeads       # auto-downloaded
        ├── images/
        └── metadata.csv       # must contain a `split` column; only `split == test` is evaluated
```

Each `metadata.csv` for a MYST subset provides the following columns:

| Column | Description |
|---|---|
| `path` | Image path, relative to the dataset folder |
| `identity` | Individual turtle ID |
| `orientation` | Annotated head yaw: `L` (left), `R` (right), `T` (top) |
| `timestamp` | Capture time from EXIF. `%d/%m/%Y %H:%M` for `mantanani`, `%Y-%m-%d %H:%M:%S` for all others |

#### Dataset Availability

> [!NOTE]
> **The MYST dataset will be released later.** It is not included in this repository yet. This section will be updated with the download link once the release is finalized.

<!-- TODO: replace the note above with the actual distribution link (e.g. a release asset, Zenodo DOI, or
     request form) once the dataset is published. -->

The source-domain dataset (`SeaTurtleIDHeads`) is public and is downloaded automatically by `wildlife-datasets`.

## Evaluation

MYST evaluates pre-trained models in a **zero-shot** setting — there is no training stage. To benchmark a model on a dataset, run:

```bash
uv run main.py model=<MODEL> dataset=<DATASET>
```

`MODEL`: Can be selected from {`wildlife`, `miewid`, `aliked`, `disk`, `sift`} \
`DATASET`: Can be selected from {`seaturtleidheads`, `aquatic`, `redang`, `mantanani`, `sipadan`, `pompom_j`, `pompom_s`}

To reproduce every row of the main benchmark table:

```bash
bash run_experiment.sh
```

### Benchmarks

The three benchmarks reported in the paper map onto these configs:

| Benchmark | Config | Generalization gap tested |
|---|---|---|
| Source Domain (Wild Loggerhead) | `dataset=seaturtleidheads` | In-domain baseline |
| MYST-Aquatic | `dataset=aquatic` | Ecological (*ex-situ* → *in-situ*) |
| MYST-Terrestrial | `dataset=redang` | Medium (aquatic → terrestrial) |

Impostor identities for each benchmark are declared per dataset config. For example
`configs/dataset/redang.yaml` draws its cross-domain (*easy*) impostors from the aquatic subsets:

```yaml
name: redang
imposter_datasets:
  - mantanani
  - pompom_j
  - pompom_s
  - sipadan
```

### Models

| Config | Model | Matcher |
|---|---|---|
| `sift` | SIFT (hand-crafted) | LightGlue |
| `disk` | DISK | LightGlue |
| `aliked` | ALIKED | LightGlue |
| `wildlife` | MegaDescriptor-L-384 (Swin-Large) | Cosine similarity |
| `miewid` | MiewID (EfficientNet) | Cosine similarity |

### Configuration Override

Configuration is managed by [Hydra](https://hydra.cc/):

```bash
# Override a single parameter
uv run main.py model=aliked dataset=redang seed=123

# Point at a different dataset root
uv run main.py model=aliked dataset=redang dataset.root=/path/to/data

# Report DIR at several FPIR operating points
uv run main.py model=aliked dataset=redang evaluation.fpir_targets=[0.01,0.05]

# Aggregate over all orientations instead of evaluating per head orientation
uv run main.py model=aliked dataset=redang evaluation=standard

# Apply CLAHE preprocessing
uv run main.py model=aliked dataset=redang model.use_clahe=true

# Sweep with Hydra multirun
uv run main.py -m model=sift,disk,aliked,wildlife,miewid dataset=redang
```

### Protocol

The **Chronological Open-Set (COS)** protocol is implemented in [`src/data.py`](src/data.py) (`DataManager.prepare_gq_split`) and [`src/evaluation.py`](src/evaluation.py):

- **Gallery** — the single earliest-timestamp image per enrolled identity (identities with ≥ 2 sightings).
- **Query (known)** — all later images of enrolled identities.
- **Query (unknown)** — singleton identities (`imposter_in`) plus, for cross-domain evaluation, every identity
  from the opposing domain (`imposter_ex`).
- **Side-specific matching** — with `evaluation=per_view` (the default), queries are only matched against
  gallery images of the same head orientation, so the asymmetry of turtle facial scutes does not confound the
  measured domain shift. Per-orientation scores are pooled into an overall score weighted by query count.

### Output Structure

Each run creates a Hydra-timestamped directory:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/                      # Hydra's resolved config
        ├── query_metadata_<L|R|T>.csv    # Per-query scores, predicted identity, genuine/impostor role
        └── outputs/
            └── YYYYMMDD_HHMMSS/
                ├── config.yaml           # Saved configuration
                ├── experiment.log        # Detailed logs
                └── overall_results.csv   # Per-orientation + overall metrics
```

### Metrics

`overall_results.csv` reports, per head orientation and pooled overall:

| Metric | Description |
|---|---|
| `CMC@1`, `CMC@5`, `CMC@10` | Rank-k accuracy on known queries (closed-set) |
| `mAP` | Mean Average Precision on known queries |
| `DIR@FPIR=0.01` | **Detection and Identification Rate at 1% FPIR** — the fraction of known queries that are both ranked first *and* score above the threshold at which 1% of impostors are accepted. Couples identification with rejection. |
| `Thresh@FPIR=0.01` | Similarity threshold yielding the target FPIR |
| `TNR@TPR` | True Negative Rate at 95% TPR — reported for comparison, and shown in the paper to *mask* high-confidence misidentifications |
| `G`, `Q(C)`, `Q(O)` | Gallery size, closed-set query count, open-set (impostor) query count |

## Results

Rank-1 (%) and DIR@1%FPIR (%). **Bold**: best, _underlined_: second best.

| | Source Domain<br>*(SeaTurtleIDHeads)* | | | Aquatic<br>*(Ecological)* | | | Terrestrial<br>*(Medium)* | | |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | Rank-1 | mAP | DIR@1% | Rank-1 | mAP | DIR@1% | Rank-1 | mAP | DIR@1% |
| *Local Descriptors* | | | | | | | | | |
| SIFT | 40.1 | 46.0 | 26.4 | _76.2_ | _81.2_ | **65.2** | 36.1 | 48.5 | 19.4 |
| DISK | 44.1 | 49.5 | 33.6 | 68.5 | 72.2 | 53.6 | 80.6 | 85.8 | 61.1 |
| ALIKED | 58.8 | 62.4 | 45.7 | 72.5 | 76.3 | 59.6 | **97.2** | **98.6** | **91.7** |
| *Global Descriptors* | | | | | | | | | |
| MegaDescriptor | **95.5** | **96.7** | **87.9** | 69.9 | 77.4 | 41.4 | 47.2 | 61.3 | 22.2 |
| MiewID | _91.1_ | _92.7_ | _78.0_ | **82.5** | **86.3** | _64.6_ | _94.4_ | _96.1_ | _58.3_ |

While MegaDescriptor dominates the source domain, it is outperformed by hand-crafted SIFT on the *ecological gap* (Aquatic) and collapses on the *medium shift* (Terrestrial), where the local descriptor ALIKED excels.

## Citation

If you find our paper and repository useful, please cite

```bibtex
@inproceedings{icip2026_myst,
  author = {Nah, Wan Jun and Joseph, Juanita and Saw, Shier Nee and Hoo, Wai Lam},
  booktitle = {2026 IEEE International Conference on Image Processing (ICIP)},
  title = {MYST: Benchmarking Ecological and Cross-Medium Generalization in Sea Turtle Re-Identification},
  year = {2026}
}
```

## Feedback

Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the
authors by sending an email to `nicolenahwj at gmail.com`.

&#169; 2026 Universiti Malaya.
