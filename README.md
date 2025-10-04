
---

# LLM-Idea-Bench: Benchmarking LLM-Based Research Assistants

Tools and scaffolding to **generate research ideas with LLM agents** and **evaluate them** against both **human/LLM judge ratings** and **literature-based novelty signals**. The repo is organized as three focused components:

* `ai_scientist/` — Agentic pipeline (AI Scientist v2) for idea generation → experiment planning → paper writing. This fork adds optional **local Ollama** model support. See folder README for setup & usage. ([GitHub][1])
* `citation_graph/` — Utilities for building/consuming bibliographic features (e.g., similarity/novelty proxies) used in evaluation. See folder README.
* `benchmark/` — A small **CLI** to merge **LLM judge JSON ratings** with **biblio novelty** CSVs, compute **summary metrics** (e.g., Spearman rank-corr, bootstrap CIs), and emit diagnostic **plots**. See folder README for examples and I/O schema. ([GitHub][2])

## Repo structure

```
llm-idea-bench/
├─ ai_scientist/        # agent system for generating ideas & papers  → see README
├─ citation_graph/      # literature graph & novelty/similarity utils → see README
└─ benchmark/           # judge+biblio merge, metrics & plots         → see README
```

## Getting started (at a glance)

1. **Environment**
   Use a recent Python (≥3.9). Create a venv and install per the component you’ll use. The `benchmark/` tool only needs `numpy`, `pandas`, `matplotlib`; the agent system has additional dependencies (PyTorch, LaTeX/PDF tooling, etc.). See the respective READMEs for exact commands. ([GitHub][2])

2. **Generate ideas (optional, via agent)**
   Run the AI Scientist v2 workflow (cloud APIs or **local models via Ollama**) to produce ideas, code, and drafts. Detailed steps and safety notes are in `ai_scientist/README.md`. ([GitHub][1])

3. **Prepare evaluation data**

* **Ratings**: Collect LLM judge outputs (one JSON per seed/topic).
* **Bibliography**: Export idea- or seed-level CSVs with similarity columns; novelty is derived as `1 - similarity` by the benchmarking tool. See exact column expectations in `benchmark/README.md`. ([GitHub][2])

4. **Run the benchmark CLI**
   Point it at your ratings directory and one or more biblio CSVs; it will write a merged table, **metrics summary**, and **plots** to an output folder (scatter, calibration by novelty bins, radar of rubric profiles, etc.). Example commands and outputs are documented in `benchmark/README.md`. ([GitHub][2])

## Results & artifacts

* `benchmark/` produces:

  * `merged.csv` (tidy rows per model/seed/idea with judge + biblio features),
  * `metrics.txt` (counts, verdicts, rank correlations, bootstrap CIs),
  * PNG plots (scatter, calibration, radar, boxplot, heatmap). See folder README for details. ([GitHub][2])

## Documentation

* **AI Scientist agent & local model support:** [`ai_scientist/README.md`](./ai_scientist/README.md) — includes attribution to the original AI Scientist v2 project and notes on Ollama usage. ([GitHub][1])
* **Benchmark CLI:** [`benchmark/README.md`](./benchmark/README.md) — inputs, schema, metrics, and quick-start examples. ([GitHub][2])
* **Citation/novelty utilities:** [`citation_graph/README.md`](./citation_graph/README.md)

## Acknowledgements

* This repository adapts and extends **AI Scientist v2** for local model support; please credit and cite the **original authors** as described in `ai_scientist/README.md`. ([GitHub][1])

## License

See component-level READMEs for any licensing notes. If absent, add a top-level `LICENSE` file appropriate for your use case.

---


[1]: https://github.com/lucapnh/llm-idea-bench/raw/main/ai_scientist/README.md "raw.githubusercontent.com"
[2]: https://github.com/lucapnh/llm-idea-bench/raw/main/benchmark/README.md "raw.githubusercontent.com"
