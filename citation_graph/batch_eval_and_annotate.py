#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests

ARXIV_RE = re.compile(r'(\d{4}\.\d{5})(?:v\d+)?')

def extract_arxiv_id(seed_label: str) -> Optional[str]:
    m = ARXIV_RE.search(seed_label)
    return m.group(1) if m else None

def fetch_title_for_arxiv(arxiv_id: str, api_key: str) -> Optional[str]:
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        resp = requests.get(url, params={"fields": "title"}, headers={"x-api-key": api_key}, timeout=(10, 30))
        if resp.status_code == 200:
            j = resp.json()
            return j.get("title")
    except Exception:
        pass
    return None

def add_seed_columns_to_summary(summary_csv: Path, seed_label: str, seed_id_full: str, seed_title: Optional[str]) -> Path:
    if not summary_csv.exists():
        print(f"[warn] summary not found: {summary_csv}", file=sys.stderr)
        return summary_csv

    tmp_out = summary_csv.with_name("summary_with_seed.csv")
    with summary_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        # Put seed columns first, keep the rest as-is
        base_fields = reader.fieldnames or []
        extra = ["seed_label", "seed_id", "seed_title"]
        # Avoid duplicate columns if the script already writes some of them
        fieldnames = extra + [f for f in base_fields if f not in extra]
        with tmp_out.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row_out = dict(row)
                row_out["seed_label"] = seed_label
                row_out["seed_id"] = seed_id_full
                row_out["seed_title"] = seed_title
                writer.writerow(row_out)
    # Atomically replace old summary
    tmp_out.replace(summary_csv)
    return summary_csv

def append_to_master(master_csv: Path, source_csv: Path):
    # Append rows from source_csv to master_csv; create master if missing
    with source_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        if not rows:
            return
        fieldnames = reader.fieldnames or []

    write_header = not master_csv.exists()
    with master_csv.open("a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    p = argparse.ArgumentParser(description="Batch-run citation_graph_sbert_enhanced_summ.py for each seed and annotate summaries.")
    p.add_argument("--logs-root", required=True, help="Root dir that contains <SeedPaperName>/...JSON logs")
    p.add_argument("--results-root", required=True, help="Where to store per-seed outputs and merged summary_all.csv")
    p.add_argument("--script", default="citation_graph_sbert_enhanced_summ.py", help="Path to evaluator script")
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--sbert-threshold", type=float, default=0.55)
    p.add_argument("--hybrid-threshold", type=float, default=0.25)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--api-key", default=os.environ.get("S2_API_KEY", ""), help="S2 API key (or set S2_API_KEY)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.api_key:
        print("Error: provide --api-key or set S2_API_KEY env var.", file=sys.stderr)
        sys.exit(2)

    logs_root = Path(args.logs_root)
    results_root = Path(args.results_root)
    script_path = Path(args.script)

    if not logs_root.is_dir():
        print(f"Error: --logs-root not a directory: {logs_root}", file=sys.stderr)
        sys.exit(2)
    results_root.mkdir(parents=True, exist_ok=True)
    if not script_path.exists():
        print(f"Error: evaluator script not found: {script_path}", file=sys.stderr)
        sys.exit(2)

    seeds = sorted([p for p in logs_root.iterdir() if p.is_dir()])
    if not seeds:
        print(f"No seed folders found under {logs_root}", file=sys.stderr)
        sys.exit(1)

    master_summary = results_root / "summary_all.csv"

    for seed_dir in seeds:
        seed_label = seed_dir.name  # e.g., AutoGen2308.08155
        arxiv_id = extract_arxiv_id(seed_label)
        if not arxiv_id:
            print(f"[skip] cannot parse arXiv id from folder: {seed_label}", file=sys.stderr)
            continue
        seed_id_full = f"arXiv:{arxiv_id}"

        out_dir = results_root / seed_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "hits.csv"
        out_summary = out_dir / "summary.csv"

        cmd = [
            sys.executable, str(script_path), "evaluate-logs",
            "--seed", seed_id_full,
            "--logs-dir", str(seed_dir),
            "--out-csv", str(out_csv),
            "--out-summary-csv", str(out_summary),
            "--alpha", str(args.alpha),
            "--sbert-threshold", str(args.sbert_threshold),
            "--hybrid-threshold", str(args.hybrid_threshold),
            "--api-key", args.api_key,
        ]
        if args.verbose:
            cmd.append("--verbose")

        print(f"[run] {seed_label} -> {seed_id_full}")
        if not args.dry_run:
            # Run evaluator
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] evaluator failed for {seed_label}: {e}", file=sys.stderr)
                continue

        # Fetch title once per seed
        seed_title = fetch_title_for_arxiv(arxiv_id, args.api_key)

        # Inject seed columns, then append to master
        updated = add_seed_columns_to_summary(out_summary, seed_label, seed_id_full, seed_title)
        append_to_master(master_summary, updated)

    print(f"\nDone. Merged summary at: {master_summary}")

if __name__ == "__main__":
    main()
