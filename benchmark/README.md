# Benchmark — README

A small command‑line tool to **merge LLM judge ratings with bibliographic similarity/novelty signals**, compute summary metrics, and produce diagnostic plots. Useful for comparing models and checking whether judged quality tracks with literature‑based novelty.



---

## What it does

- Parses a directory of **LLM judge JSON files** (one per seed/topic) and extracts:
  - `model` (heuristically inferred from file path),
  - seed title, idea IDs/titles,
  - per‑rubric subscores,
  - `overall_weighted_score`,
  - optional `verdict`.
- Loads one or more **bibliography CSVs** (idea‑level and/or seed‑level) and coerces them to a consistent schema with a **biblio novelty** column in [0,1].
- **Merges** judges + biblio on `(model, seed_title_slug, idea_id_slug)`, computes an **effective novelty** (idea‑level when available, else seed proxy), and writes tidy outputs. 
- Computes summary **metrics** per model (Spearman rank‑corr, bootstrap CIs, verdict counts) and saves a set of **plots** to help you eyeball calibration and agreement across models. 

---

## Installation

```bash
# Python 3.9+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib
```

No other external deps are required. The script uses only the Python standard library + the packages above. 

---

## Quick start

The following example mirrors the command in your evaluation notes:

```bash
python benchmark.py \
  --ratings_dir /evaluation/ratings \
  --biblio_csv /evaluation/biblio_summary/summary_llama3.1.csv \
               /evaluation/biblio_summary/summary_all_deepseekv3.csv \
  --out_dir ./eval_out_v2
```

(Above paths are examples from `evaluation_notes.txt`.) 

---

## CLI

```text
python benchmark.py --ratings_dir <DIR> --biblio_csv <CSV...> --out_dir <DIR>
```

- `--ratings_dir` (required): directory with JSON ratings (recursively scanned). 
- `--biblio_csv` (1+ required): one or more CSV files with biblio features (idea‑level and/or seed‑level). 
- `--out_dir` (required): destination for merged table, metrics, and plots. 

---

## Inputs

### 1) LLM judge JSONs

Each JSON is expected to include a seed/topic and a list of ideas. The loader is fault‑tolerant and extracts the following when present:  
- `seed_summary.title` or the first `seed_summaries[].title` → used for **seed slugging**.  
- `ideas[]` list where each item may contain:
  - `idea_id`/`Name`/`name` (fallbacks) and `title`/`Title`,
  - `scores` dict → becomes `score_<rubric>` numeric columns,
  - `overall_weighted_score` (float),
  - `verdict` (string). 

Model name is **guessed from the path** (tokens like `llama3.1`, `deepseekv3`, etc.; otherwise the parent folder name or `unknown`). 

### 2) Bibliography CSVs

Provide **either** (preferably both):

- **Idea‑level CSV** — must contain an idea identifier column (any of: `idea_id`, `Name`, `name`, `idea`, `idea_name`, `idea_slug`, `Title`, `title`). Optional seed title column (any of: `seed_title`, `seed`, `seed_name`, `topic_title`, `topic`). One similarity column is detected from likely names (e.g., `hybrid_similarity`, `reference_similarity_time_decayed_jaccard`, `sbert_mean`, `similarity`, etc.). The script converts similarity `s` to **novelty** as `1 - clip01(s)` into `biblio_novelty`. 

- **Seed‑level CSV** — must contain a seed title column (as above). Novelty is derived from available medians/maxes, e.g. `biblio_novelty_seed = 1 - hybrid_median` (or `1 - sbert_median`; falls back to `hybrid_max`/`sbert_max`). 

Both CSV types get normalized to lowercase columns; a `model` column is added if missing (inferred from path). Slugs are derived from `seed_title` and `idea_id`. 

---

## Outputs (in `--out_dir`)

- `merged.csv` — row per (seed, idea, model) with judge fields + biblio features and  
  `biblio_novelty_effective` = `biblio_novelty` **or** (if missing) `biblio_novelty_seed`. 
- `metrics.txt` — per‑model summary including:
  - counts,  
  - `novelty_col` actually used,  
  - `Spearman(judge, novelty)` (rank correlation),  
  - `Mean judge overall` with **bootstrap 95% CI**,  
  - `Mean novelty` with CI,  
  - verdict distribution. fileciteturn0file0
- Plots (PNG, timestamped filenames):
  - **Scatter**: judge vs. effective novelty, per model.  
  - **Calibration**: mean judge score across novelty quantile bins.  
  - **Radar**: mean rubric profile (`score_*`) per model.  
  - **Boxplot**: overall score distribution by model.  
  - **Heatmap**: pairwise Spearman rank correlation of models.

> Plots are saved with a non‑clashing timestamp suffix.

---

## How things are matched

Rows are merged on:
- `(model, seed_title_slug, idea_id_slug)` for idea‑level CSVs, and
- `(model, seed_title_slug)` for seed‑level summaries. 

String slugs are lowercased, whitespace→dashes, and stripped of non‑alphanumerics.

---

## Notes on metrics

- **Spearman rank correlation** between judge overall and the chosen novelty column.  
- **Bootstrap mean CI** (default 3,000 resamples, α=0.05).

---

## Troubleshooting

- **“No usable biblio novelty found”**  
  Provide at least one of:
  - an **idea‑level** CSV with a recognizable similarity column, or
  - a **seed‑level** CSV with `hybrid_median` / `sbert_median` (or `*_max`). 

- **Low match rate after merge**  
  Check that `seed_title` and idea identifiers in CSVs align with the judge JSON titles/IDs (after slugging). Consider adding an explicit `model` column to the CSVs or renaming files/parent folders so the `path_model_guess` finds the intended token. 

- **Empty plots / NaNs**  
  The plotting functions require at least a few valid points per model (e.g., ≥3 for correlations). Ensure your inputs contain numeric `overall_weighted_score` and similarity values. 

---

## Repro tips

- Keep consistent seed titles across pipelines that generate judges and biblio data.  
- Prefer **idea‑level** CSVs when available; seed‑level novelty is a coarse proxy.  
- Version your input folders; the script timestamps plot filenames for you. 

---

## Example project layout

```
evaluation/
  ratings/                       # JSON files from LLM judges (any depth)
  biblio_summary/
    summary_llama3.1.csv
    summary_all_deepseekv3.csv
outputs/
```

Run:

```bash
python benchmark.py \
  --ratings_dir evaluation/ratings \
  --biblio_csv evaluation/biblio_summary/summary_llama3.1.csv \
               evaluation/biblio_summary/summary_all_deepseekv3.csv \
  --out_dir outputs
```

---

## License

MIT (or your preferred license).
