# Citation Graph + SBERT — README

A small toolkit to compare scientific papers by their citation neighborhoods and (optionally) abstract semantics using SBERT. It also includes utilities to evaluate Semantic Scholar search logs in batch and to enrich/merge results.

---

## What this does (at a glance)

* Fetches **references** and **citations** for papers from the Semantic Scholar Graph API and computes:

  * Jaccard, Salton cosine, and **time-decayed Jaccard** similarities over reference and co-citation sets.
  * A **hybrid** score (weighted blend of reference vs. co-citation similarity). 
* (Optional) Computes **SBERT cosine similarity** between paper abstracts. 
* CLI subcommands:

  * `compare` — compare two papers.
  * `evaluate-logs` — score top results in a folder of Semantic Scholar search log JSONs against a **seed** paper.
  * `resolve` — map a free-text title to a Semantic Scholar paperId.
  * `compare-batch` — compare a seed against a list of titles/ids from stdin or a file. 
* A helper script `batch_eval.py` to run `evaluate-logs` across many seeds and produce a merged summary. 

---

## Requirements

* Python 3.8+

* Dependencies:

  * Required: `requests`, `python-dateutil`
  * Optional (to enable SBERT abstracts): `transformers>=4.40`, `torch`
    Install via:

  ```bash
  pip install requests python-dateutil
  # SBERT (optional)
  pip install "transformers>=4.40" torch
  ```

   

* Semantic Scholar API key in the environment:

  ```bash
  export S2_API_KEY='YOUR_KEY'
  ```

  (or provide `--api-key` to commands.)  

---

## Install / Setup

This is a simple script repo; you can run it in-place:

```bash
python -m venv .venv
source .venv/bin/activate
pip install requests python-dateutil
pip install "transformers>=4.40" torch  # optional
export S2_API_KEY='YOUR_KEY'
```

The main entry points are `citation_graph.py` and `batch_eval.py`.  

---

## Quickstart

### 1) Compare two papers

Compare by DOI/arXiv/S2 ID/URL:

```bash
python citation_graph.py compare \
  --paper1 arXiv:2308.08155 \
  --paper2 10.1145/3366423.3380105 \
  --alpha 0.6 \
  --half-life 5.0 \
  --sbert-model sentence-transformers/all-MiniLM-L6-v2 \
  --api-key "$S2_API_KEY" \
  --verbose
```

* `alpha` ∈ [0,1]: weight of reference similarity vs. co-citation similarity in the **hybrid** score.
* `half-life`: years for time decay in the time-decayed Jaccard (newer links weigh more).
* Use `--no-sbert` to disable SBERT even if dependencies are installed.
* Add `--export-prefix out/myrun` to write CSVs of **shared references** and **shared citations**. 

The command prints a JSON result with:

* `paper1` / `paper2` metadata (title, year) and abstracts (if available),
* `counts` of refs/citations and overlaps,
* `reference_similarity`, `co_citation_similarity`,
* `hybrid_similarity` (components + final `score`),
* `abstract_semantic_similarity` (SBERT status and `cosine` if enabled),
* `exported_files` (paths to CSVs if exported). 

### 2) Evaluate Semantic Scholar logs against a seed

Given a folder of `*.json` files (each with a `query` and a `"result": {"1": "...", "2": "...", ...}` map), score the top results vs. a **seed** paper:

```bash
python citation_graph.py evaluate-logs \
  --seed arXiv:2308.08155 \
  --logs-dir /path/to/semantic_scholar_logs \
  --out-csv hits.csv \
  --out-summary-csv summary.csv \
  --alpha 0.6 \
  --sbert-threshold 0.55 \
  --hybrid-threshold 0.25 \
  --api-key "$S2_API_KEY" \
  --verbose
```

* Produces per-hit `hits.csv` and a per-log `summary.csv` with median/min/max hybrid scores and novelty fractions (below thresholds), plus SBERT stats if available. 
* A concrete example of the above flow is also shown in `citation_graph_notes.txt`. 

### 3) Batch across many seeds and merge summaries

Use `batch_eval.py` to iterate seeds (directories under `--logs-root`) and create a **merged** `summary_all.csv`:

```bash
python batch_eval.py \
  --logs-root   /path/to/semantic_scholar_logs \
  --results-root /path/to/batchresults \
  --script /path/to/citation_graph.py \
  --alpha 0.6 --sbert-threshold 0.55 --hybrid-threshold 0.25 \
  --api-key "$S2_API_KEY" \
  --verbose
```

What it does:

* For each seed folder (e.g., `AutoGen2308.08155`), extracts the arXiv id, runs `evaluate-logs`, and writes per-seed outputs.
* Fetches the canonical **title** for the seed via Graph API and **injects** `seed_label`, `seed_id`, `seed_title` columns into each seed’s `summary.csv`.
* Appends everything into a master `summary_all.csv` under `--results-root`. 

> Tip: `--script` can point to any compatible evaluator (defaults to `citation_graph_sbert_enhanced_summ.py` in the notes, but `citation_graph.py`’s `evaluate-logs` works the same).  

---

## CLI Reference

### `compare`

```
usage: citation_graph.py compare --paper1 P1 --paper2 P2
                                [--alpha F] [--half-life Y] [--current-year N]
                                [--api-key KEY] [--verbose] [--max-retries N]
                                [--limit N] [--page-size N] [--sleep-after-page S]
                                [--export-prefix PATH] [--out FILE]
                                [--sbert-model NAME] [--no-sbert]
```

Key options: reference/citation limits & paging; CSV export of overlaps; JSON output via `--out`. 

### `evaluate-logs`

```
usage: citation_graph.py evaluate-logs --seed ID --logs-dir DIR
                                       [--api-key KEY] [--alpha F] [--half-life Y]
                                       [--max-per-log K] [--out-csv PATH] [--out-summary-csv PATH]
                                       [--sbert-threshold T] [--hybrid-threshold T]
                                       [--model-hint NAME] [--limit N] [--page-size N] [--verbose]
```

Resolves top titles per log to paperIds, runs `compare`, and writes per-hit & summary CSVs with novelty fractions. 

### `resolve`

```
usage: citation_graph.py resolve --title "Paper title"
```

Returns `{paperId, title, year}` for the best match. 

### `compare-batch`

```
usage: citation_graph.py compare-batch --seed ID [--input FILE]
```

Reads one title/id per line (or stdin), resolves if needed, and writes `compare_batch_hits.csv` with `input, paperId, title, year, hybrid_similarity`. 

---

## Output files

### JSON (from `compare`)

* `counts` — refs/cites sizes and overlaps.
* `reference_similarity` & `co_citation_similarity` — `{jaccard, salton_cosine, time_decayed_jaccard}`.
* `hybrid_similarity` — `{alpha, half_life_years, score, components:{refs, cites}}`.
* `abstract_semantic_similarity` — `{enabled, model, cosine, paper1_missing, paper2_missing}`.
* `exported_files` — `{shared_refs_csv, shared_citations_csv}` if `--export-prefix` used. 

### CSV (from `evaluate-logs`)

* **Per-hit** CSV columns:
  `model, log_file, query_hint, rank, paperId, title, year, hybrid_similarity, sbert_cosine, sbert_model`. 
* **Per-log summary** CSV columns include:
  `n_hits, hybrid_median, hybrid_min, hybrid_max, novel_fraction_hybrid<TH>, n_sbert, sbert_median, sbert_min, sbert_max, novel_fraction_sbert<TH>, joint_fraction_novel`. 

### CSV (from `--export-prefix` on `compare`)

* `*_shared_refs.csv`, `*_shared_citations.csv` with columns: `paperId,title,year`. 

---

## Similarity details

* **Time-decayed Jaccard**: weights items by age using a half-life (default 5 years), so recent overlaps matter more. You can also clamp by `--current-year`. 
* **Hybrid score**: `alpha * ref_sim + (1 - alpha) * co_citation_sim`. Set `alpha` closer to 1.0 to focus on shared references; closer to 0.0 to emphasize co-citations. 
* **SBERT**: Uses `sentence-transformers/all-MiniLM-L6-v2` by default if `transformers`/`torch` are available; otherwise disabled gracefully. 

---

## Robustness & Rate Limits

* The Semantic Scholar API is called with retries/backoff for transient errors (429/5xx). `--max-retries` controls the retry budget. Respect rate limits and consider `--sleep-after-page` for large fetches. 

---

## Project structure

```
citation_graph.py      # main CLI/library: compare, evaluate-logs, resolve, compare-batch
batch_eval.py          # multi-seed batch runner + merged summary_all.csv
citation_graph_notes.txt  # quick commands & environment notes
```

  

---

## Examples

* From the notes:

  ```bash
  python citation_graph.py evaluate-logs \
    --seed arXiv:2308.08155 \
    --logs-dir /pfss/.../semantic_scholar_logs \
    --out-csv hits.csv --out-summary-csv summary.csv \
    --alpha 0.6 --sbert-threshold 0.55 --hybrid-threshold 0.25 \
    --api-key $S2_API_KEY --verbose
  ```

  ```bash
  python batch_eval.py \
    --logs-root /pfss/.../semantic_scholar_logs \
    --results-root /pfss/.../batchresults \
    --script /pfss/.../citation_graph_sbert_enhanced_summ.py \
    --alpha 0.6 --sbert-threshold 0.55 --hybrid-threshold 0.25 --verbose
  ```



---

## Troubleshooting

* **“Missing API key” / 401** — Set `S2_API_KEY` or pass `--api-key`. 
* **SBERT disabled** — If `transformers`/`torch` aren’t installed, SBERT is skipped automatically; install them or add `--no-sbert`. 
* **Long runs** — Lower `--limit` or increase `--page-size`, and consider `--sleep-after-page` for politeness. 

---

## Security & Keys

Never commit your API key. Prefer an environment variable (`S2_API_KEY`) or a secrets manager; the CLI accepts `--api-key` for ad-hoc usage. 

---

## License

Specify your license here (e.g., MIT). If you add one, include the `LICENSE` file.

---

## Citation

If you use this tool in a paper or project, please include a link to this repository and credit the Semantic Scholar Graph API.

---

## Acknowledgements

Built on the Semantic Scholar Graph API and (optionally) SBERT (via Hugging Face Transformers). 

---

**Happy paper-matching!**
