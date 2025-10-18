from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from dateutil.tz import tzutc

#API setup

API_BASE = "https://api.semanticscholar.org/graph/v1"
DEFAULT_TIMEOUT = (10, 30)  # (connect, read) seconds
HEADERS_BASE = {
    "Accept": "application/json",
    "User-Agent": "citation-graph/1.4-sbert-enhanced (+https://github.com/you/yourtool)",
}

def now_utc_year() -> int:
    return datetime.now(tzutc()).year

#Utilities

def log(msg: str, *, verbose: bool):
    if verbose:
        print(msg, file=sys.stderr)

def clamp_year(y: Optional[int], current_year: int) -> Optional[int]:
    if y is None:
        return None
    return min(y, current_year)

_S2_HEX_RE = re.compile(r"([0-9a-f]{40})(?:[\\?#/]|$)", re.IGNORECASE)

def extract_s2_id_from_url(url: str) -> Optional[str]:
    m = _S2_HEX_RE.search(url.strip())
    return m.group(1).lower() if m else None

def normalize_paper_identifier(s: str) -> str:
    s = s.strip()
    if s.lower().startswith(("http://", "https://")):
        s2 = extract_s2_id_from_url(s)
        return s2 if s2 else s
    return s

TITLE_RE = re.compile(r"^(.*?)[\\.\n]")

def extract_title(raw_entry: str) -> str:
    m = TITLE_RE.search(raw_entry.strip())
    if m:
        title = m.group(1).strip()
    else:
        title = raw_entry.strip()
    title = re.sub(r"\\s+", " ", title)
    return title

def ensure_dir_for(path_prefix: str):
    d = os.path.dirname(os.path.abspath(path_prefix))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

#HTTP / API layer

class S2Client:
    def __init__(self, api_key: str, verbose: bool, max_retries: int):
        self.api_key = api_key
        self.verbose = verbose
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(HEADERS_BASE)
        if api_key:
            self.session.headers["x-api-key"] = api_key

    def _request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        import random, time
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, timeout=DEFAULT_TIMEOUT, **kwargs)
            except requests.RequestException as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(f"Network error for {url} after {self.max_retries} retries: {e}") from e
                log(f"[warn] Network error: {e}. Backing off (attempt {attempt})...", verbose=self.verbose)
                time.sleep(min(60.0, (2 ** attempt)) + random.uniform(0.05, 0.35) * (attempt + 1))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return {}

            if resp.status_code in (429, 502, 503, 504):
                attempt += 1
                if attempt > self.max_retries:
                    try:
                        j = resp.json()
                    except Exception:
                        j = {"message": resp.text}
                    raise RuntimeError(
                        f"API error {resp.status_code} for {url} after {self.max_retries} retries: {j}"
                    )
                ra = resp.headers.get("Retry-After")
                retry_after = None
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
                if retry_after:
                    time.sleep(min(120.0, max(0.0, retry_after)))
                else:
                    import random
                    time.sleep(min(60.0, (2 ** attempt)) + random.uniform(0.05, 0.35) * (attempt + 1))
                continue

            try:
                j = resp.json()
            except Exception:
                j = {"message": resp.text}
            if resp.status_code == 404:
                raise RuntimeError(f"Not found (404) for {url}. Details: {j}")
            raise RuntimeError(f"API error {resp.status_code} for {url}: {j}")

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{API_BASE}{path}"
        return self._request("GET", url, params=params)

    def post_json(self, path: str, json_body: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{API_BASE}{path}"
        return self._request("POST", url, json=json_body, params=params)

#API helpers

def page_items(
    client: S2Client,
    paper_id: str,
    endpoint: str,  
    fields: str,
    *, limit: int = 1000, page_size: int = 100, sleep_after_page: float = 0.0,
) -> List[Dict[str, Any]]:
    import time
    items: List[Dict[str, Any]] = []
    offset = 0
    while True:
        params = {"fields": fields, "limit": page_size, "offset": offset}
        data = client.get(f"/paper/{paper_id}/{endpoint}", params)
        batch = data.get("data", []) or []
        items.extend(batch)
        log(f"[info] fetched {len(batch)} {endpoint} (total {len(items)})", verbose=client.verbose)
        if len(batch) < page_size or len(items) >= limit:
            break
        offset += page_size
        if sleep_after_page > 0:
            time.sleep(sleep_after_page)
    return items[:limit]

def get_paper_minimal(client: S2Client, paper_id: str) -> Dict[str, Any]:
    params = {"fields": "paperId,title,year,externalIds"}
    return client.get(f"/paper/{paper_id}", params)

def get_paper_with_abstract(client: S2Client, paper_id: str) -> Dict[str, Any]:
    params = {"fields": "paperId,title,year,externalIds,abstract"}
    return client.get(f"/paper/{paper_id}", params)

def get_references(client: S2Client, paper_id: str, *, limit: int = 3000, page_size: int = 100, sleep_after_page: float = 0.0) -> List[Dict[str, Any]]:
    fields = "citedPaper.paperId,citedPaper.title,citedPaper.year,contexts,intents,isInfluential,year"
    return page_items(client, paper_id, "references", fields, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page)

def get_citations(client: S2Client, paper_id: str, *, limit: int = 3000, page_size: int = 100, sleep_after_page: float = 0.0) -> List[Dict[str, Any]]:
    fields = "citingPaper.paperId,citingPaper.title,citingPaper.year,contexts,intents,isInfluential,year"
    return page_items(client, paper_id, "citations", fields, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page)

def search_title_one(client: S2Client, title: str) -> Tuple[str, Dict[str, Any]]:
    """Return (paperId, meta) best match for title, or ('', {}) if none."""
    data = client.get("/paper/search", {"query": title, "limit": 1, "fields": "paperId,title,year,externalIds"})
    papers = data.get("data", []) if isinstance(data, dict) else []
    if papers:
        p = papers[0]
        return p.get("paperId") or "", p
    return "", {}

#Similarity metrics

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union) if union else 0.0

def salton_cosine(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / ((len(a) * len(b)) ** 0.5)

def time_decayed_jaccard(a_items: Dict[str, Optional[int]], b_items: Dict[str, Optional[int]], *, half_life_years: float = 5.0, current_year: Optional[int] = None) -> float:
    if current_year is None:
        current_year = now_utc_year()
    keys = set(a_items.keys()) | set(b_items.keys())
    if not keys:
        return 0.0
    def wt(y: Optional[int]) -> float:
        if y is None:
            return 1.0
        y = min(y, current_year)
        age = max(0.0, current_year - y)
        return 0.5 ** (age / half_life_years)
    inter_sum = 0.0
    union_sum = 0.0
    for k in keys:
        wa = wt(a_items.get(k))
        wb = wt(b_items.get(k))
        inter_sum += min(wa, wb)
        union_sum += max(wa, wb)
    return inter_sum / union_sum if union_sum > 0 else 0.0

#Data shaping

def build_meta_map_from_references(refs: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[Any]]]:
    out: Dict[str, Dict[str, Optional[Any]]] = {}
    for r in refs:
        cp = r.get("citedPaper") or {}
        pid = cp.get("paperId")
        if pid:
            out[pid] = {"title": cp.get("title"), "year": cp.get("year")}
    return out

def build_meta_map_from_citations(cites: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[Any]]]:
    out: Dict[str, Dict[str, Optional[Any]]] = {}
    for c in cites:
        cp = c.get("citingPaper") or {}
        pid = cp.get("paperId")
        if pid:
            out[pid] = {"title": cp.get("title"), "year": cp.get("year")}
    return out

#Batch enrichment for titles/years

def fetch_titles_for_ids(client: S2Client, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}
    enriched: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        data = client.post_json("/paper/batch", json_body={"ids": batch}, params={"fields": "paperId,title,year"})
        if isinstance(data, list):
            for p in data:
                if not p:
                    continue
                pid = p.get("paperId")
                if pid:
                    enriched[pid] = {"title": p.get("title"), "year": p.get("year")}
        else:
            papers = data.get("data", []) if isinstance(data, dict) else []
            for p in papers:
                pid = p.get("paperId")
                if pid:
                    enriched[pid] = {"title": p.get("title"), "year": p.get("year")}
    return enriched

#SBERT

@dataclass
class SbertContext:
    model_name: str
    tokenizer: Any
    model: Any
    device: str

def maybe_load_sbert(model_name: str, *, verbose: bool) -> Optional[SbertContext]:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel  # type: ignore
    except Exception as e:
        log(f"[info] SBERT disabled: transformers/torch not available ({e})", verbose=verbose)
        return None
    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    log(f"[info] Loading SBERT model '{model_name}' on {device}...", verbose=verbose)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return SbertContext(model_name=model_name, tokenizer=tokenizer, model=model, device=device)

def _mean_pool(last_hidden_state, attention_mask):
    import torch
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def sbert_embed_texts(ctx: SbertContext, texts: List[str]):
    import torch
    enc = ctx.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(ctx.device) for k, v in enc.items()}
    with torch.no_grad():
        out = ctx.model(**enc)
    embs = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs

def sbert_cosine(ctx: Optional[SbertContext], a: Optional[str], b: Optional[str], *, verbose: bool) -> Dict[str, Any]:
    res = {"enabled": False, "model": None, "cosine": None, "paper1_missing": a is None or a.strip() == "", "paper2_missing": b is None or b.strip() == ""}
    if ctx is None:
        return res
    res["enabled"] = True
    res["model"] = ctx.model_name
    if res["paper1_missing"] or res["paper2_missing"]:
        return res
    embs = sbert_embed_texts(ctx, [a, b])
    import torch
    cos = float(torch.matmul(embs[0], embs[1]).item())
    res["cosine"] = max(0.0, min(1.0, cos))
    return res

#Core compare

def compare_papers(
    paper1: str,
    paper2: str,
    *,
    api_key: str,
    alpha: float = 0.5,
    decay_half_life: float = 5.0,
    verbose: bool = False,
    max_retries: int = 6,
    limit: int = 3000,
    page_size: int = 100,
    sleep_after_page: float = 0.0,
    export_prefix: Optional[str] = None,
    current_year: Optional[int] = None,
    sbert_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    enable_sbert: bool = True,
) -> Dict[str, Any]:
    p1 = normalize_paper_identifier(paper1)
    p2 = normalize_paper_identifier(paper2)
    client = S2Client(api_key=api_key, verbose=verbose, max_retries=max_retries)

    # Fetch minimal + abstract for both
    meta1 = get_paper_with_abstract(client, p1)
    meta2 = get_paper_with_abstract(client, p2)

    id1 = meta1.get("paperId") or p1
    id2 = meta2.get("paperId") or p2

    need_refs = True
    need_cits = (alpha < 1.0) or (export_prefix is not None)

    refs1 = get_references(client, id1, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page) if need_refs else []
    refs2 = get_references(client, id2, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page) if need_refs else []
    cits1: List[Dict[str, Any]] = []
    cits2: List[Dict[str, Any]] = []
    if need_cits:
        cits1 = get_citations(client, id1, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page)
        cits2 = get_citations(client, id2, limit=limit, page_size=page_size, sleep_after_page=sleep_after_page)

    ref_meta1 = build_meta_map_from_references(refs1)
    ref_meta2 = build_meta_map_from_references(refs2)
    cit_meta1 = build_meta_map_from_citations(cits1)
    cit_meta2 = build_meta_map_from_citations(cits2)

    ref_set1, ref_set2 = set(ref_meta1.keys()), set(ref_meta2.keys())
    cit_set1, cit_set2 = set(cit_meta1.keys()), set(cit_meta2.keys())

    shared_refs = ref_set1 & ref_set2
    shared_cits = cit_set1 & cit_set2

    # Reference similarities
    ref_jacc = jaccard(ref_set1, ref_set2)
    ref_cos = salton_cosine(ref_set1, ref_set2)
    ref_tj = time_decayed_jaccard(
        {k: (ref_meta1.get(k) or {}).get("year") for k in ref_meta1},
        {k: (ref_meta2.get(k) or {}).get("year") for k in ref_meta2},
        half_life_years=decay_half_life,
        current_year=current_year,
    )

    # Citation similarities
    if need_cits:
        cit_jacc = jaccard(cit_set1, cit_set2)
        cit_cos = salton_cosine(cit_set1, cit_set2)
        cit_tj = time_decayed_jaccard(
            {k: (cit_meta1.get(k) or {}).get("year") for k in cit_meta1},
            {k: (cit_meta2.get(k) or {}).get("year") for k in cit_meta2},
            half_life_years=decay_half_life,
            current_year=current_year,
        )
    else:
        cit_jacc = cit_cos = cit_tj = 0.0

    ref_sim = ref_tj if (ref_set1 or ref_set2) else 0.0
    cit_sim = cit_tj if (cit_set1 or cit_set2) else 0.0
    hybrid = alpha * ref_sim + (1 - alpha) * cit_sim

    # SBERT abstracts
    sbert_ctx = None
    if enable_sbert and sbert_model:
        sbert_ctx = maybe_load_sbert(sbert_model, verbose=verbose)
    abs1 = (meta1.get("abstract") or None)
    abs2 = (meta2.get("abstract") or None)
    abstract_sem = sbert_cosine(sbert_ctx, abs1, abs2, verbose=verbose)

    exported: Dict[str, str] = {}
    if export_prefix:
        refs_path = f"{export_prefix}_shared_refs.csv"
        cits_path = f"{export_prefix}_shared_citations.csv"

        ref_info = {**ref_meta1, **ref_meta2}
        cit_info = {**cit_meta1, **cit_meta2}

        need_ref_enrich = [pid for pid in shared_refs if not (ref_info.get(pid, {}).get("title"))]
        need_cit_enrich = [pid for pid in shared_cits if not (cit_info.get(pid, {}).get("title"))]

        if need_ref_enrich:
            log(f"[info] enriching {len(need_ref_enrich)} shared ref IDs via /paper/batch", verbose=verbose)
            ref_enriched = fetch_titles_for_ids(client, need_ref_enrich)
            for pid, meta in ref_enriched.items():
                ref_info[pid] = {**ref_info.get(pid, {}), **meta}

        if need_cit_enrich:
            log(f"[info] enriching {len(need_cit_enrich)} shared citation IDs via /paper/batch", verbose=verbose)
            cit_enriched = fetch_titles_for_ids(client, need_cit_enrich)
            for pid, meta in cit_enriched.items():
                cit_info[pid] = {**cit_info.get(pid, {}), **meta}

        write_overlap_csv(refs_path, rows_from_intersection(shared_refs, ref_info))
        write_overlap_csv(cits_path, rows_from_intersection(shared_cits, cit_info))
        exported = {"shared_refs_csv": refs_path, "shared_citations_csv": cits_path}

    return {
        "paper1": {"paperId": id1, "title": meta1.get("title"), "year": meta1.get("year"), "abstract": abs1},
        "paper2": {"paperId": id2, "title": meta2.get("title"), "year": meta2.get("year"), "abstract": abs2},
        "counts": {
            "refs_p1": len(ref_set1), "refs_p2": len(ref_set2),
            "cits_p1": len(cit_set1), "cits_p2": len(cit_set2),
            "shared_refs": len(shared_refs),
            "shared_cits": len(shared_cits),
        },
        "reference_similarity": {"jaccard": ref_jacc, "salton_cosine": ref_cos, "time_decayed_jaccard": ref_tj},
        "co_citation_similarity": {"jaccard": cit_jacc, "salton_cosine": cit_cos, "time_decayed_jaccard": cit_tj},
        "hybrid_similarity": {"alpha": alpha, "half_life_years": decay_half_life, "score": hybrid, "components": {"refs": ref_sim, "cites": cit_sim}},
        "abstract_semantic_similarity": abstract_sem,
        "exported_files": exported
    }

#CSV helpers

def write_overlap_csv(path: str, rows: List[Dict[str, Any]]):
    ensure_dir_for(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paperId", "title", "year"])
        w.writeheader()
        for r in rows:
            w.writerow({"paperId": r.get("paperId"), "title": r.get("title"), "year": r.get("year")})

def rows_from_intersection(shared_ids: Set[str], meta_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pid in sorted(shared_ids):
        meta = meta_map.get(pid, {})
        rows.append({"paperId": pid, "title": meta.get("title"), "year": meta.get("year")})
    return rows

#Evaluate logs

def infer_idea_hint_from_filename(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^\\d{8}_\\d{6}_", "", name)
    return name

def resolve_title_to_paper_id(api_key: str, title: str, *, verbose: bool = False, max_retries: int = 4) -> Tuple[str, Dict[str, Any]]:
    client = S2Client(api_key=api_key, verbose=verbose, max_retries=max_retries)
    return search_title_one(client, title)

def evaluate_logs(
    seed: str,
    logs_dir: str,
    api_key: str,
    *,
    alpha: float = 0.6,
    half_life: float = 5.0,
    max_per_log: int = 5,
    out_csv: str = "results_hits.csv",
    out_summary_csv: str = "results_summary.csv",
    model_hint: Optional[str] = None,
    limit: int = 2000,
    page_size: int = 100,
    verbose: bool = False,
    sbert_threshold: float = 0.55,
    hybrid_threshold: float = 0.25,
) -> Tuple[str, str]:
    client = S2Client(api_key=api_key, verbose=verbose, max_retries=6)

    hit_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    logs_path = Path(logs_dir).expanduser().resolve()
    files = sorted([p for p in logs_path.glob("*.json") if p.is_file()])

    for lf in files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log(f"[warn] Skipping {lf.name}: cannot read JSON ({e})", verbose=verbose)
            continue

        query = data.get("query") or infer_idea_hint_from_filename(lf)
        results_map: Dict[str, str] = (data.get("result") or {})
        try:
            keys = sorted(results_map.keys(), key=lambda k: int(k))
        except Exception:
            keys = list(results_map.keys())

        per_log_scores: List[float] = []
        per_log_sbert: List[float] = []
        for rank, k in enumerate(keys[:max_per_log], start=1):
            raw_entry = results_map.get(k, "")
            title = extract_title(raw_entry) if raw_entry else ""
            if not title:
                continue

            pid, pmeta = search_title_one(client, title)
            if not pid:
                log(f"[warn] Could not resolve paperId for: {title}", verbose=verbose)
                continue

            try:
                comp = compare_papers(
                    seed, pid,
                    api_key=api_key,
                    alpha=alpha,
                    decay_half_life=half_life,
                    verbose=verbose,
                    limit=limit,
                    page_size=page_size,
                )
                hybrid = comp.get("hybrid_similarity", {}).get("score")
                sbert_meta = comp.get("abstract_semantic_similarity", {}) or {}
                sbert_cos = sbert_meta.get("cosine")
                sbert_model_used = sbert_meta.get("model")
            except Exception as e:
                log(f"[warn] compare_papers failed for {pid}: {e}", verbose=verbose)
                hybrid = None
                sbert_cos = None
                sbert_model_used = None

            row = {
                "model": model_hint or Path(logs_dir).name,
                "log_file": lf.name,
                "query_hint": query,
                "rank": rank,
                "paperId": pid,
                "title": pmeta.get("title"),
                "year": pmeta.get("year"),
                "hybrid_similarity": hybrid,
                "sbert_cosine": sbert_cos,
                "sbert_model": sbert_model_used
            }
            hit_rows.append(row)
            if hybrid is not None:
                per_log_scores.append(float(hybrid))
            # Track SBERT values if available
            per_log_sbert = locals().get('per_log_sbert', None)
            if per_log_sbert is None:
                per_log_sbert = []
                locals()['per_log_sbert'] = per_log_sbert
            if sbert_cos is not None:
                try:
                    per_log_sbert.append(float(sbert_cos))
                except Exception:
                    pass

        if per_log_scores:
            med = sorted(per_log_scores)[len(per_log_scores)//2]
            sbert_stats = {
                "sbert_median": (sorted(per_log_sbert)[len(per_log_sbert)//2] if per_log_sbert else None),
                "sbert_min": (min(per_log_sbert) if per_log_sbert else None),
                "sbert_max": (max(per_log_sbert) if per_log_sbert else None),
            }
            novel_hybrid = sum(1 for s in per_log_scores if s < hybrid_threshold) / len(per_log_scores)
            novel_sbert = (sum(1 for s in per_log_sbert if s < sbert_threshold) / len(per_log_sbert)) if per_log_sbert else None
            # Joint novel fraction where both metrics indicate novelty
            joint_novel = None
            if per_log_sbert:
                pairs = list(zip(per_log_scores, per_log_sbert))
                joint_novel = sum(1 for h, sc in pairs if (h is not None and sc is not None and h < hybrid_threshold and sc < sbert_threshold)) / len(pairs)
            summary_rows.append({
                "model": model_hint or Path(logs_dir).name,
                "log_file": lf.name,
                "query_hint": query,
                "n_hits": len(per_log_scores),
                "hybrid_median": med,
                "hybrid_min": min(per_log_scores),
                "hybrid_max": max(per_log_scores),
                "novel_fraction_hybrid<{}".format(hybrid_threshold): novel_hybrid,
                "n_sbert": (len(per_log_sbert) if per_log_sbert else 0),
                **sbert_stats,
                "novel_fraction_sbert<{}".format(sbert_threshold): novel_sbert,
                "joint_fraction_novel": joint_novel
            })
        else:
            summary_rows.append({
                "model": model_hint or Path(logs_dir).name,
                "log_file": lf.name,
                "query_hint": query,
                "n_hits": 0,
                "hybrid_median": None,
                "hybrid_min": None,
                "hybrid_max": None,
                "novel_fraction_hybrid<{}".format(hybrid_threshold): None,
                "n_sbert": 0,
                "sbert_median": None,
                "sbert_min": None,
                "sbert_max": None,
                "novel_fraction_sbert<{}".format(sbert_threshold): None,
                "joint_fraction_novel": None
            })

    #write CSVs
    def write_csv(path: str, rows: List[Dict[str, Any]]):
        ensure_dir_for(path)
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out_csv, hit_rows)
    write_csv(out_summary_csv, summary_rows)
    return out_csv, out_summary_csv

#CLI

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Citation graph + SBERT + batch evaluation tool.")
    sub = p.add_subparsers(dest="cmd")

    # compare (default) - keep original flags
    pc = sub.add_parser("compare", help="Compare two papers (refs/cites + SBERT).")
    pc.add_argument("--paper1", required=True)
    pc.add_argument("--paper2", required=True)
    pc.add_argument("--alpha", type=float, default=0.5)
    pc.add_argument("--half-life", type=float, default=5.0)
    pc.add_argument("--current-year", type=int, default=None)
    pc.add_argument("--api-key", default=os.environ.get("S2_API_KEY", ""))
    pc.add_argument("--verbose", action="store_true")
    pc.add_argument("--max-retries", type=int, default=6)
    pc.add_argument("--limit", type=int, default=3000)
    pc.add_argument("--page-size", type=int, default=100)
    pc.add_argument("--sleep-after-page", type=float, default=0.0)
    pc.add_argument("--export-prefix", default=None)
    pc.add_argument("--out", default=None)
    pc.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    pc.add_argument("--no-sbert", action="store_true")

    # evaluate-logs
    pl = sub.add_parser("evaluate-logs", help="Evaluate a folder of Semantic Scholar logs against a SEED.")
    pl.add_argument("--seed", required=True, help="Seed paper identifier (DOI:..., arXiv:..., S2 ID or URL).")
    pl.add_argument("--logs-dir", required=True, help="Directory with *.json logs.")
    pl.add_argument("--api-key", default=os.environ.get("S2_API_KEY", ""))
    pl.add_argument("--alpha", type=float, default=0.6)
    pl.add_argument("--half-life", type=float, default=5.0)
    pl.add_argument("--max-per-log", type=int, default=5)
    pl.add_argument("--out-csv", default="results_hits.csv")
    pl.add_argument("--out-summary-csv", default="results_summary.csv")
    pl.add_argument("--sbert-threshold", type=float, default=0.55)
    pl.add_argument("--hybrid-threshold", type=float, default=0.25)
    pl.add_argument("--model-hint", default=None)
    pl.add_argument("--limit", type=int, default=2000)
    pl.add_argument("--page-size", type=int, default=100)
    pl.add_argument("--verbose", action="store_true")

    # resolve
    pr = sub.add_parser("resolve", help="Resolve a title to a Semantic Scholar paperId.")
    pr.add_argument("--title", required=True)
    pr.add_argument("--api-key", default=os.environ.get("S2_API_KEY", ""))
    pr.add_argument("--verbose", action="store_true")

    # compare-batch
    pb = sub.add_parser("compare-batch", help="Compare a seed against many titles/ids from stdin or file.")
    pb.add_argument("--seed", required=True)
    pb.add_argument("--api-key", default=os.environ.get("S2_API_KEY", ""))
    pb.add_argument("--input", default="-", help="Path to file with one title/id per line, or '-' for stdin.")
    pb.add_argument("--alpha", type=float, default=0.6)
    pb.add_argument("--half-life", type=float, default=5.0)
    pb.add_argument("--limit", type=int, default=2000)
    pb.add_argument("--page-size", type=int, default=100)
    pb.add_argument("--verbose", action="store_true")
    pb.add_argument("--out-csv", default="compare_batch_hits.csv")

    return p

def main():
    p = build_arg_parser()
    args = p.parse_args()
    cmd = args.cmd or "compare"

    # Common sanity checks
    if cmd in ("compare", "evaluate-logs", "resolve", "compare-batch"):
        api_key = getattr(args, "api_key", os.environ.get("S2_API_KEY", ""))
        if not api_key:
            print("Error: Missing API key. Provide --api-key or set S2_API_KEY.", file=sys.stderr)
            sys.exit(2)

    if cmd == "compare":
        try:
            result = compare_papers(
                args.paper1,
                args.paper2,
                api_key=args.api_key,
                alpha=args.alpha,
                decay_half_life=args.half_life,
                verbose=args.verbose,
                max_retries=args.max_retries,
                limit=args.limit,
                page_size=args.page_size,
                sleep_after_page=args.sleep_after_page,
                export_prefix=args.export_prefix,
                current_year=args.current_year,
                sbert_model=(None if args.no_sbert else args.sbert_model),
                enable_sbert=(not args.no_sbert),
            )
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr); sys.exit(130)
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr); sys.exit(1)

        payload = json.dumps(result, indent=2, ensure_ascii=False)
        if args.out:
            try:
                ensure_dir_for(args.out)
                with open(args.out, "w", encoding="utf-8") as f:
                    f.write(payload)
            except Exception as e:
                print(f"Error writing --out file: {e}", file=sys.stderr); sys.exit(1)
        else:
            print(payload)
        return

    if cmd == "evaluate-logs":
        try:
            hits, summary = evaluate_logs(
                args.seed,
                args.logs_dir,
                args.api_key,
                alpha=args.alpha,
                half_life=args.half_life,
                max_per_log=args.max_per_log,
                out_csv=args.out_csv,
                out_summary_csv=args.out_summary_csv,
                model_hint=args.model_hint,
                limit=args.limit,
                page_size=args.page_size,
                verbose=args.verbose,
                sbert_threshold=args.sbert_threshold,
                hybrid_threshold=args.hybrid_threshold,
            )
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr); sys.exit(130)
        except Exception as e:
            print(f"Failed: {e}", file=sys.stderr); sys.exit(1)
        print(f"Wrote per-hit CSV: {hits}")
        print(f"Wrote per-log summary CSV: {summary}")
        return

    if cmd == "resolve":
        pid, meta = resolve_title_to_paper_id(args.api_key, args.title, verbose=args.verbose)
        print(json.dumps({"paperId": pid, "title": meta.get("title"), "year": meta.get("year")}, indent=2))
        return

    if cmd == "compare-batch":
        # read lines
        def source_lines():
            if args.input == "-":
                for line in sys.stdin:
                    yield line.strip()
            else:
                with open(args.input, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line.strip()

        rows: List[Dict[str, Any]] = []
        for line in source_lines():
            if not line:
                continue
            title_or_id = line
            if not re.search(r"^[0-9a-f]{40}$", title_or_id) and not title_or_id.lower().startswith(("doi:", "arxiv:", "http://", "https://")):
                pid, meta = resolve_title_to_paper_id(args.api_key, title_or_id, verbose=args.verbose)
                if not pid:
                    continue
            else:
                pid, meta = title_or_id, {"title": None, "year": None}

            try:
                comp = compare_papers(
                    args.seed, pid,
                    api_key=args.api_key,
                    alpha=args.alpha,
                    decay_half_life=args.half_life,
                    verbose=args.verbose,
                    limit=args.limit,
                    page_size=args.page_size,
                )
                hybrid = comp.get("hybrid_similarity", {}).get("score")
            except Exception as e:
                hybrid = None

            rows.append({
                "input": line,
                "paperId": pid,
                "title": meta.get("title"),
                "year": meta.get("year"),
                "hybrid_similarity": hybrid
            })

        ensure_dir_for(args.out_csv)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["input", "paperId", "title", "year", "hybrid_similarity"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote: {args.out_csv}")
        return

if __name__ == "__main__":
    main()
