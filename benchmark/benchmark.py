
# !/usr/bin/env python3
import argparse, json, math, os, re, sys, glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import re 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def slug(s):
    if s is None: return None
    s = re.sub(r"\\s+", "-", str(s).strip().lower())
    s = re.sub(r"[^a-z0-9\\-]+", "", s)
    return s or None

def path_model_guess(path):
    base = path.lower()
    tokens = ["llama3.1","llama-3.1","deepseekv3","deepseek-671b","deepseek-671","deepseekv2","236b","405b","671b"]
    for t in tokens:
        if t in base: return t
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else "unknown"

def nowstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v): return v
    except Exception:
        pass
    return np.nan

def pick_first(df, cols):
    for c in cols:
        if c in df.columns: return c
    return None

def prettify_label(name: str) -> str:
    if not isinstance(name, str): return str(name)
    raw = name.strip().lower().replace("-", "_")
    corrections = {
        "literature_grounding": "literature_grounding",
        "clarity_specificity": "clarity_specificity",
        "safety_ethics_risk": "safety_ethics_risk",
        "feasibility_resources": "feasibility_resources",
        "testability_falsifiability": "testability_falsifiability",
        "relevance_alignment": "relevance_alignment",
        "methodological_rigor": "methodological_rigor",
        "originality_novelty": "originality_novelty",
        "potential_impact": "potential_impact",
    }
    if raw in corrections:
        return corrections[raw]
    words = [w for w in re.split(r"[_\\s]+", raw) if w]
    pretty = " ".join(words).title()
    pretty = re.sub(r"\\bLlm\\b", "LLM", pretty)
    return pretty

def break_ligatures(s: str) -> str:
    # Insert zero-width non-joiners to break 'fi', 'fl', 'ffi', 'ffl' ligatures
    if not isinstance(s, str): return s
    z = "\u200C"  # ZWNJ
    s = s.replace("ffi", f"f{z}f{z}i")
    s = s.replace("ffl", f"f{z}f{z}l")
    s = s.replace("fi", f"f{z}i")
    s = s.replace("fl", f"f{z}l")
    return s

# ---------------- load LLM judge JSON ----------------
def load_llm_judges(ratings_dir):
    rows = []
    for path in glob.glob(os.path.join(ratings_dir, "**", "*.json"), recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        model = path_model_guess(path)
        seed_title = None
        if isinstance(data.get("seed_summary"), dict):
            seed_title = data["seed_summary"].get("title")
        if not seed_title and isinstance(data.get("seed_summaries"), list) and data["seed_summaries"]:
            seed_title = data["seed_summaries"][0].get("title")
        seed_slug = slug(seed_title) if seed_title else None
        ideas = data.get("ideas") or []
        for iobj in ideas:
            idea_id = iobj.get("idea_id") or iobj.get("Name") or iobj.get("name") or None
            title = iobj.get("title") or iobj.get("Title") or None
            scores = iobj.get("scores") or {}
            overall = iobj.get("overall_weighted_score")
            verdict = iobj.get("verdict")
            rows.append({
                "source_file": os.path.basename(path),
                "model": model,
                "seed_title": seed_title,
                "seed_title_slug": seed_slug,
                "idea_id": idea_id if idea_id is not None else title,
                "idea_id_slug": slug(idea_id) if idea_id else slug(title),
                "idea_title": title,
                "overall_weighted_score": safe_float(overall),
                "verdict": verdict,
                **{f"score_{k}": safe_float(v) for k,v in scores.items()}
            })
    return pd.DataFrame(rows)

IDEA_KEYS = ["idea_id","Name","name","idea","idea_name","idea_slug","Title","title"]
SEED_KEYS = ["seed_title","seed","seed_name","topic_title","topic"]
LIKELY_SIM_COLS = [
    "hybrid_similarity","hybrid","hybrid_score",
    "reference_similarity_time_decayed_jaccard","ref_time_decayed_jaccard","time_decayed_jaccard",
    "co_citation_similarity_time_decayed_jaccard","cocite_time_decayed_jaccard",
    "ref_jaccard","ref_cosine","cit_jaccard","cit_cosine",
    "similarity","sbert_similarity","sbert_mean","mean_similarity"
]

def coerce_biblio_idea_level(df, model_hint):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "model" not in df.columns:
        df["model"] = model_hint
    idea_col = pick_first(df, [c.lower() for c in IDEA_KEYS])
    if idea_col is None:
        raise ValueError("no idea identifier")
    df["idea_id"] = df[idea_col].astype(str)
    df["idea_id_slug"] = df["idea_id"].map(slug)
    seed_col = pick_first(df, [c.lower() for c in SEED_KEYS])
    df["seed_title"] = df[seed_col].astype(str) if seed_col else None
    df["seed_title_slug"] = df["seed_title"].map(slug) if seed_col else None
    sim_col = pick_first(df, LIKELY_SIM_COLS)
    df["biblio_similarity"] = pd.to_numeric(df[sim_col], errors="coerce") if sim_col else np.nan
    df["biblio_novelty"] = 1.0 - df["biblio_similarity"].clip(0,1)
    keep = ["model","seed_title","seed_title_slug","idea_id","idea_id_slug","biblio_similarity","biblio_novelty"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[sorted(set(keep + numeric_cols))]
    group_key = ["model","seed_title_slug","idea_id_slug"]
    agg_map = {c: "first" for c in ["seed_title","idea_id"] if c in df.columns}
    for c in df.columns:
        if c not in group_key and c not in agg_map:
            if pd.api.types.is_numeric_dtype(df[c]):
                agg_map[c] = "mean"
            else:
                agg_map[c] = "first"
    return (df.sort_values(["seed_title","idea_id"], na_position="last")
              .groupby(group_key, as_index=False)
              .agg(agg_map))

def coerce_biblio_seed_level(df, model_hint):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "model" not in df.columns:
        df["model"] = model_hint
    seed_col = pick_first(df, [c.lower() for c in SEED_KEYS])
    if seed_col is None:
        raise ValueError("seed-level CSV without seed_title")
    df["seed_title"] = df[seed_col].astype(str)
    df["seed_title_slug"] = df["seed_title"].map(slug)
    def clip01(x): return pd.to_numeric(x, errors="coerce").clip(0,1)
    if "hybrid_median" in df.columns:
        df["biblio_novelty_seed"] = 1 - clip01(df["hybrid_median"])
    elif "sbert_median" in df.columns:
        df["biblio_novelty_seed"] = 1 - clip01(df["sbert_median"])
    else:
        for c in ["hybrid_max","sbert_max"]:
            if c in df.columns:
                df["biblio_novelty_seed"] = 1 - clip01(df[c]); break
    keep = ["model","seed_title","seed_title_slug","biblio_novelty_seed"]
    extra = [c for c in df.columns if c.startswith(("hybrid_","sbert_","novel_fraction","joint_fraction"))]
    df = df[sorted(set(keep + extra))]
    group_key = ["model","seed_title_slug"]
    agg_map = {c: "first" for c in ["seed_title"] if c in df.columns}
    for c in df.columns:
        if c not in group_key and c not in agg_map:
            if pd.api.types.is_numeric_dtype(df[c]):
                agg_map[c] = "mean"
            else:
                agg_map[c] = "first"
    return (df.sort_values(["seed_title"], na_position="last")
              .groupby(group_key, as_index=False)
              .agg(agg_map))

def load_biblio(csv_paths):
    idea_frames, seed_frames = [], []
    for p in csv_paths:
        model_hint = path_model_guess(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}", file=sys.stderr); continue
        cols_lower = [c.strip().lower() for c in df.columns]
        has_idea_key = any(c in cols_lower for c in [c.lower() for c in IDEA_KEYS])
        has_seed_title = any(c in cols_lower for c in [c.lower() for c in SEED_KEYS])
        try:
            if has_idea_key:
                idea_frames.append(coerce_biblio_idea_level(df, model_hint))
            elif has_seed_title:
                seed_frames.append(coerce_biblio_seed_level(df, model_hint))
            else:
                print(f"[warn] {p}: neither idea-level nor seed-level schema recognized.", file=sys.stderr)
        except Exception as e:
            print(f"[warn] {p}: {e}", file=sys.stderr)
    df_idea = pd.concat(idea_frames, ignore_index=True) if idea_frames else pd.DataFrame()
    df_seed = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    if not df_idea.empty:
        df_idea = (df_idea.groupby(["model","seed_title_slug","idea_id_slug"], as_index=False)
                          .agg({**{c:"first" for c in ["seed_title","idea_id"] if c in df_idea.columns},
                                **{c:"mean" for c in df_idea.select_dtypes(include=[np.number]).columns}}))
    if not df_seed.empty:
        df_seed = (df_seed.groupby(["model","seed_title_slug"], as_index=False)
                          .agg({**{c:"first" for c in ["seed_title"] if c in df_seed.columns},
                                **{c:"mean" for c in df_seed.select_dtypes(include=[np.number]).columns}}))
    return df_idea, df_seed

def spearman(a, b):
    s = pd.Series(a); t = pd.Series(b)
    m = s.notna() & t.notna()
    if m.sum() < 3: return np.nan
    return s[m].rank().corr(t[m].rank(), method="spearman")

def bootstrap_mean_ci(x, n=3000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return (float(np.mean(x)), float(lo), float(hi))

def savefig_noclash(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{nowstamp()}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path

def choose_novelty_col(df):
    if "biblio_novelty" in df.columns:
        return "biblio_novelty"
    if "biblio_novelty_seed" in df.columns:
        return "biblio_novelty_seed"
    return None

def plot_scatter(df, out_dir):
    nov_col = choose_novelty_col(df)
    if not nov_col:
        return

    model_name_map = {
        "deepseekv3": "Deepseek-v3:671B",
        "deepseekv2": "Deepseek-v2:236B",
        "llama3.1": "LLaMA-3.1:405B",
    }

    for model, g in df.groupby("model"):
        plt.figure()
        m = g["overall_weighted_score"].notna() & g[nov_col].notna()
        if m.sum() < 3:
            plt.close()
            continue

        plt.scatter(g.loc[m, nov_col], g.loc[m, "overall_weighted_score"], alpha=0.7)
        plt.xlabel("Biblio novelty (idea-level)" if nov_col == "biblio_novelty" else "Biblio novelty (seed proxy)")
        plt.ylabel("LLM judge overall")
        r = spearman(g[nov_col], g["overall_weighted_score"])

        ax = plt.gca()
        ax.set_xlim(0.20, 0.55)
        ax.set_ylim(2, 5)
        ax.set_xticks([0.25, 0.35, 0.45, 0.55])
        ax.set_yticks([2.5, 3.0, 3.5, 4.0, 4.5])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis="both", which="major", labelsize=22)

        pretty_name = model_name_map.get(str(model).lower(), str(model))

        plt.title(f"{pretty_name} (Spearman ρ={r:.2f}, n={int(m.sum())})")
        savefig_noclash(out_dir, f"scatter_judge_vs_{nov_col}_{slug(pretty_name)}")


def plot_calibration(df, out_dir, bins=5):
    nov_col = choose_novelty_col(df)
    if not nov_col: return
    for model, g in df.groupby("model"):
        plt.figure()
        m = g["overall_weighted_score"].notna() & g[nov_col].notna()
        if m.sum() < bins: plt.close(); continue
        q = pd.qcut(g.loc[m,nov_col], q=bins, duplicates="drop")
        cal = g.loc[m].groupby(q)["overall_weighted_score"].mean()
        idx = np.arange(len(cal))
        plt.plot(idx, cal.values, marker="o")
        plt.xticks(idx, [f"{a.left:.2f}-{a.right:.2f}" for a in cal.index], rotation=25, ha="right")
        plt.xlabel("Biblio novelty bins")
        plt.ylabel("Mean judge score")
        plt.title(f"Calibration — {model}")
        savefig_noclash(out_dir, f"calibration_{nov_col}_{slug(model)}")

def plot_radar_subscores(df, out_dir):
    subcols = [c for c in df.columns if c.startswith("score_")]
    if not subcols: return
    raw_labels = [c.replace("score_","") for c in subcols]
    labels = [prettify_label(x) for x in raw_labels]
    # Break ligatures to prevent 'fi'/'fl' disappearing
    labels = [break_ligatures(x) for x in labels]
    angles = np.linspace(0, 2*np.pi, len(subcols), endpoint=False).tolist()
    angles += angles[:1]
    for model, g in df.groupby("model"):
        means = [g[c].mean() for c in subcols]
        if all(np.isnan(means)): continue
        vals = means + means[:1]
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
            tick.set_fontfamily("DejaVu Sans")
            tick.set_fontsize(14)
        ax.tick_params(pad=8)
        rmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
        rings = 5
        rticks = np.linspace(0, rmax, rings + 1)[1:]
        ax.set_ylim(0, rmax)
        ax.set_yticks(rticks)
        ax.set_yticklabels([""] * len(rticks))
        ax.set_xticklabels(labels, fontsize=20, fontweight="normal", fontfamily="DejaVu Sans")
        ax.grid(True, alpha=0.5, linewidth=0.8)
        #ax.set_title(f"Rubric Profile — {str(model).capitalize()}", fontsize=20)
        savefig_noclash(out_dir, f"radar_subscores_{slug(model)}")


def _cap_model_name(s: str) -> str:
    # Uppercase any lowercase letter at the start or after a non-letter
    return re.sub(r'(^|[^A-Za-z])([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)

def plot_model_box(df, out_dir):
    plt.figure()
    g = df.dropna(subset=["overall_weighted_score"])
    if g.empty:
        plt.close()
        return

    order = g.groupby("model")["overall_weighted_score"].mean().sort_values(ascending=False).index.tolist()
    data = [g.loc[g["model"] == m, "overall_weighted_score"].values for m in order]
    if not data:
        plt.close()
        return

    # Map model IDs to display names
    name_map = {
        "deepseekv3": "DeepSeek-v3:671B",
        "deepseekv2": "DeepSeek-v2:236B",
        "llama3.1": "LLaMA-3.1:405B"
    }

    # Use mapped names, fallback to original if not found
    labels = [name_map.get(m.lower(), m) for m in order]

    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("LLM judge overall")
    plt.title("Model score distribution (overall)")
    savefig_noclash(out_dir, "box_overall_by_model")



def plot_pairwise_rankcorr(df, out_dir):
    key = ["seed_title_slug","idea_id_slug"]
    if any(k not in df.columns for k in key): return
    pivot = df.dropna(subset=["overall_weighted_score"]).pivot_table(
        index=key, columns="model", values="overall_weighted_score", aggfunc="mean"
    )
    models = pivot.columns.tolist()
    if len(models) < 2: return
    mat = np.zeros((len(models), len(models))) * np.nan
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                mat[i,j] = 1.0
            else:
                s = pivot[m1]; t = pivot[m2]
                m = s.notna() & t.notna()
                if m.sum() < 3: mat[i,j] = np.nan
                else: mat[i,j] = s[m].rank().corr(t[m].rank(), method="spearman")
    plt.figure()
    im = plt.imshow(mat, vmin=-1, vmax=1, aspect="auto")
    plt.xticks(range(len(models)), models, rotation=35, ha="right")
    plt.yticks(range(len(models)), models)
    cbar = plt.colorbar(im, label="Spearman ρ")
    cbar.ax.tick_params(labelsize=14)
    plt.title("Pairwise rank correlation of judge scores")
    savefig_noclash(out_dir, "heatmap_rankcorr_models")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings_dir", required=True)
    ap.add_argument("--biblio_csv", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print("[i] Loading LLM judge ratings…", file=sys.stderr)
    df_j = load_llm_judges(args.ratings_dir)
    idea_key = ["model","seed_title_slug","idea_id_slug"]
    unique_ideas = df_j[idea_key].drop_duplicates().shape[0]
    print(f"[i] Parsed {len(df_j)} idea rows from {df_j['source_file'].nunique()} JSON files "
          f"({unique_ideas} unique ideas).", file=sys.stderr)

    print("[i] Loading bibliographic CSV(s)…", file=sys.stderr)
    df_b_idea, df_b_seed = load_biblio(args.biblio_csv)
    print(f"[i] Biblio idea-level rows: {len(df_b_idea)}; seed-level rows: {len(df_b_seed)}", file=sys.stderr)

    merged = df_j.copy()

    if not df_b_idea.empty:
        dupe_mask = df_b_idea.duplicated(subset=["model","seed_title_slug","idea_id_slug"], keep=False)
        if dupe_mask.any():
            print(f"[warn] Idea-level biblio still has duplicates on merge keys; collapsing again.", file=sys.stderr)
            df_b_idea = (df_b_idea.groupby(["model","seed_title_slug","idea_id_slug"], as_index=False)
                                   .agg({**{c:"first" for c in ["seed_title","idea_id"] if c in df_b_idea.columns},
                                         **{c:"mean" for c in df_b_idea.select_dtypes(include=[np.number]).columns}}))

        merged = pd.merge(
            merged, df_b_idea,
            on=["seed_title_slug","idea_id_slug","model"],
            how="left", suffixes=("", "_b")
        )

    if not df_b_seed.empty:
        dupe_mask = df_b_seed.duplicated(subset=["model","seed_title_slug"], keep=False)
        if dupe_mask.any():
            print(f"[warn] Seed-level biblio still has duplicates on merge keys; collapsing again.", file=sys.stderr)
            df_b_seed = (df_b_seed.groupby(["model","seed_title_slug"], as_index=False)
                                   .agg({**{c:"first" for c in ["seed_title"] if c in df_b_seed.columns},
                                         **{c:"mean" for c in df_b_seed.select_dtypes(include=[np.number]).columns}}))

        cols_keep = ["model","seed_title_slug","biblio_novelty_seed"]
        extra_cols = [c for c in df_b_seed.columns if c.startswith(("hybrid_","sbert_","novel_fraction","joint_fraction"))]
        merged = pd.merge(
            merged, df_b_seed[cols_keep + extra_cols],
            on=["model","seed_title_slug"], how="left", suffixes=("", "_seed")
        )

    if ("biblio_novelty" not in merged.columns) and ("biblio_novelty_seed" not in merged.columns):
        raise SystemExit("No usable biblio novelty found. Provide idea-level CSV or seed-level summary with hybrid/sbert medians.")

    if "biblio_novelty" in merged.columns:
        nov_series = merged["biblio_novelty"]
    else:
        nov_series = pd.Series([np.nan]*len(merged), index=merged.index, name="biblio_novelty")

    if "biblio_novelty_seed" in merged.columns:
        seed_series = merged["biblio_novelty_seed"]
    else:
        seed_series = pd.Series([np.nan]*len(merged), index=merged.index, name="biblio_novelty_seed")

    merged["biblio_novelty_effective"] = nov_series.combine_first(seed_series)

    merged_path = os.path.join(args.out_dir, "merged.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[✓] Wrote {merged_path}", file=sys.stderr)

    out_lines = []
    for model, g in merged.groupby("model"):
        n_unique_ideas = g[["seed_title_slug","idea_id_slug"]].drop_duplicates().shape[0]
        n_scored = g.dropna(subset=["overall_weighted_score"])[["seed_title_slug","idea_id_slug"]].drop_duplicates().shape[0]

        nov_col = "biblio_novelty" if ("biblio_novelty" in g.columns and g["biblio_novelty"].notna().any()) else "biblio_novelty_seed"
        tmp = g.copy()
        tmp = tmp.drop_duplicates(subset=["seed_title_slug","idea_id_slug"], keep="first")
        r = np.nan
        if nov_col in tmp.columns:
            r = spearman(tmp[nov_col], tmp["overall_weighted_score"])

        mu, lo, hi = bootstrap_mean_ci(tmp["overall_weighted_score"])
        if nov_col in tmp.columns:
            mu_nov, lo_nov, hi_nov = bootstrap_mean_ci(tmp[nov_col])
        else:
            mu_nov = lo_nov = hi_nov = np.nan

        verdicts = tmp["verdict"].value_counts(dropna=False).to_dict()
        out_lines += [
            f"=== {model} ===",
            f"ideas_total={n_unique_ideas}, judge_scored={n_scored}",
            f"novelty_col={nov_col}, Spearman(judge, {nov_col}) = {r:.3f}",
            f"Mean judge overall = {mu:.3f} [{lo:.3f},{hi:.3f}]",
            f"Mean {nov_col} = {mu_nov:.3f} [{lo_nov:.3f},{hi_nov:.3f}]",
            f"Verdicts: {verdicts}",
            ""
        ]

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"[✓] Wrote {metrics_path}", file=sys.stderr)

    df_plot = merged.rename(columns={"biblio_novelty_effective":"biblio_novelty"})
    plot_scatter(df_plot, args.out_dir)
    plot_calibration(df_plot, args.out_dir)
    plot_radar_subscores(merged, args.out_dir)
    plot_model_box(merged, args.out_dir)
    plot_pairwise_rankcorr(merged, args.out_dir)
    print("[✓] Plots saved.", file=sys.stderr)

if __name__ == "__main__":
    main()
