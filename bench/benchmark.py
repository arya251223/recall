"""Benchmark the semantic cache on the labelled workload.

Usage:  python -m bench.benchmark [--seeds 5] [--llm-latency-ms 800] [--gemini]

Measured: hit rate, hit precision/recall, false hits, refusals, injection blocking, local
pipeline latency, estimated tokens.  *Modelled*: end-to-end latency, which adds an assumed
LLM round-trip (``--llm-latency-ms``) to every model call; use ``--gemini`` with a key to
replace the assumption with real calls.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.workload import HARD_NEGATIVE_FAMILIES, INJECTIONS, INTENTS, OUT_OF_SCOPE  # noqa: E402
from recall.config import Settings  # noqa: E402
from recall.metrics import Metrics  # noqa: E402
from recall.pipeline import build_pipeline  # noqa: E402

TUNE = ["pricing_pro", "pricing_business", "pricing_free", "refund_annual", "refund_monthly",
        "api_rate", "api_upload", "sso", "encryption"]
TEST = [i for i in INTENTS if i not in TUNE]  # held out: never used to choose the threshold
TEXT2INTENT = {q: i for i, qs in INTENTS.items() for q in qs}
FAMILY = {a: b for a, b in HARD_NEGATIVE_FAMILIES} | {b: a for a, b in HARD_NEGATIVE_FAMILIES}
CONFIGS = {
    "threshold-only (baseline)": dict(entity=False, consistency=False),
    "+ entity guard": dict(entity=True, consistency=False),
    "+ retrieval consistency": dict(entity=False, consistency=True),
    "full (both guards)": dict(entity=True, consistency=True),
}


def workload(seed: int, intents):
    qs = [(q, "answer") for i in intents for q in INTENTS[i]]
    qs += [(q, "oos") for q in OUT_OF_SCOPE] + [(q, "inject") for q in INJECTIONS]
    random.Random(seed).shuffle(qs)
    return qs


def run_once(p, seed: int, threshold: float, entity: bool, consistency: bool, llm_ms: float, intents):
    p.cache.clear()
    p.cache.threshold, p.cache.entity_guard, p.cfg.consistency_check = threshold, entity, consistency
    p.metrics = Metrics()
    seen_intents, r = set(), dict(hits=0, correct=0, wrong=0, wrong_family=0, ideal=0, llm=0, oos_refused=0,
                                  oos_total=0, oos_hit=0, inj_blocked=0, inj_total=0, tokens_saved=0, tokens_spent=0,
                                  lat_hit=[], lat_llm=[], modelled=[], wrong_examples=[])
    for q, kind in workload(seed, intents):
        res = p.query(q)
        if kind == "answer":
            intent = TEXT2INTENT[q]
            if intent in seen_intents:
                r["ideal"] += 1
            seen_intents.add(intent)
            if res["source"] == "cache":
                r["hits"] += 1
                got = TEXT2INTENT.get(res["matched_query"])
                if got == intent:
                    r["correct"] += 1
                else:
                    r["wrong"] += 1
                    r["wrong_family"] += FAMILY.get(intent) == got
                    r["wrong_examples"].append({"asked": q, "served_answer_for": res["matched_query"], "sim": res["similarity"]})
                r["lat_hit"].append(res["total_ms"])
                r["modelled"].append(res["total_ms"])
            elif res["source"] == "llm":
                r["llm"] += 1
                r["lat_llm"].append(res["total_ms"])
                r["modelled"].append(res["total_ms"] + llm_ms)
            r["tokens_saved"] += res["tokens"]["saved"]
            r["tokens_spent"] += res["tokens"]["prompt"] + res["tokens"]["completion"]
        elif kind == "oos":
            r["oos_total"] += 1
            r["oos_refused"] += res["source"] == "no_context"
            r["oos_hit"] += res["source"] == "cache"
        else:
            r["inj_total"] += 1
            r["inj_blocked"] += res["source"] == "blocked"
    return r


def agg(runs, key):
    return [x[key] for x in runs]


def summarise(runs, n_answerable):
    m = lambda k: st.mean(agg(runs, k))
    hits, correct = m("hits"), m("correct")
    lat_hit = [x for r in runs for x in r["lat_hit"]]
    lat_llm = [x for r in runs for x in r["lat_llm"]]
    modelled = [x for r in runs for x in r["modelled"]]
    return {
        "answerable": n_answerable,
        "hit_rate": hits / n_answerable,
        "correct_hits": correct,
        "wrong_hits": m("wrong"),
        "wrong_hits_in_hard_negative_family": m("wrong_family"),
        "hit_precision": correct / hits if hits else None,  # undefined when nothing was served from cache
        "recall_of_possible_hits": correct / m("ideal"),
        "llm_calls": m("llm"),
        "llm_calls_baseline_no_cache": n_answerable,
        "llm_call_reduction": 1 - m("llm") / n_answerable,
        "oos_refused_rate": m("oos_refused") / runs[0]["oos_total"],
        "oos_cache_hits": m("oos_hit"),
        "injection_blocked_rate": m("inj_blocked") / runs[0]["inj_total"],
        "tokens_saved_est": m("tokens_saved"),
        "tokens_spent_est": m("tokens_spent"),
        "token_reduction_est": m("tokens_saved") / max(1, m("tokens_saved") + m("tokens_spent")),
        "measured_ms_hit_mean": st.mean(lat_hit) if lat_hit else None,
        "measured_ms_hit_p95": sorted(lat_hit)[int(0.95 * (len(lat_hit) - 1))] if lat_hit else None,
        "measured_ms_llm_path_mean_excl_llm": st.mean(lat_llm) if lat_llm else None,
        "modelled_e2e_ms_mean_with_cache": st.mean(modelled) if modelled else None,
    }


def sweep(p, intents, cfg, seeds, llm_ms):
    rows = []
    n = sum(len(INTENTS[i]) for i in intents)
    for thr in [round(0.30 + 0.05 * i, 2) for i in range(14)]:
        runs = [run_once(p, sd, thr, cfg["entity"], cfg["consistency"], llm_ms, intents) for sd in range(seeds)]
        sm = summarise(runs, n)
        rows.append({"threshold": thr, **{k: sm[k] for k in ("hit_rate", "hit_precision", "recall_of_possible_hits", "wrong_hits", "oos_cache_hits")}})
    return rows


def pick_threshold(rows, min_precision=0.95):
    """Highest recall subject to precision >= min_precision (ties -> higher threshold)."""
    ok = [r for r in rows if r["hit_precision"] is not None and r["hit_precision"] >= min_precision and r["hit_rate"] > 0]
    return max(ok, key=lambda r: (r["recall_of_possible_hits"], r["threshold"]))["threshold"] if ok else 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--llm-latency-ms", type=float, default=800.0)
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.05, help="safety margin added to the tuned threshold")
    ap.add_argument("--gemini", action="store_true", help="use live Gemini (needs GEMINI_API_KEY)")
    ap.add_argument("--out", default=str(ROOT / "docs"))
    a = ap.parse_args()

    s = Settings()
    if not a.gemini:
        s.gemini_api_key = ""
    p = build_pipeline(s)
    full, naive = CONFIGS["full (both guards)"], CONFIGS["threshold-only (baseline)"]
    n_test = sum(len(INTENTS[i]) for i in TEST)

    # 1) choose thresholds on the TUNE split only
    tune_full = sweep(p, TUNE, full, a.seeds, a.llm_latency_ms)
    tune_naive = sweep(p, TUNE, naive, a.seeds, a.llm_latency_ms)
    # +margin: the tuning-split optimum is optimistic, so step one grid cell more conservative.
    thr_full = min(0.95, round(pick_threshold(tune_full, a.min_precision) + a.margin, 2))
    thr_naive = min(0.95, round(pick_threshold(tune_naive, a.min_precision) + a.margin, 2))

    # 2) report on the held-out TEST split
    out = {"embedder": p.embedder.name, "llm": p.llm.model, "kb_docs": len(p.kb.docs),
           "tune_intents": len(TUNE), "test_intents": len(TEST), "test_queries_answerable": n_test,
           "queries_out_of_scope": len(OUT_OF_SCOPE), "queries_injection": len(INJECTIONS), "seeds": a.seeds,
           "assumed_llm_latency_ms": a.llm_latency_ms, "min_precision_target": a.min_precision, "threshold_margin": a.margin,
           "threshold_full": thr_full, "threshold_naive": thr_naive, "test": {}, "test_sweep": {}, "tune_sweep": {},
           "naive_false_hit_examples": []}
    plan = [("threshold-only @ its own safe threshold", naive, thr_naive)] + [(n + f" @ {thr_full:.2f}", c, thr_full) for n, c in CONFIGS.items()]
    for name, cfg, thr in plan:
        runs = [run_once(p, sd, thr, cfg["entity"], cfg["consistency"], a.llm_latency_ms, TEST) for sd in range(a.seeds)]
        out["test"][name] = {"threshold": thr, **summarise(runs, n_test)}
        if name.startswith("threshold-only (baseline)"):
            seen, ex = set(), []
            for r_ in runs:
                for e_ in r_["wrong_examples"]:
                    if (e_["asked"], e_["served_answer_for"]) not in seen:
                        seen.add((e_["asked"], e_["served_answer_for"]))
                        ex.append(e_)
            out["naive_false_hit_examples"] = ex[:6]
    out["tune_sweep"] = {"threshold-only": tune_naive, "full": tune_full}
    out["test_sweep"] = {"threshold-only": sweep(p, TEST, naive, a.seeds, a.llm_latency_ms),
                         "full": sweep(p, TEST, full, a.seeds, a.llm_latency_ms)}

    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "benchmark_results.json").write_text(json.dumps(out, indent=2))
    plot(out, Path(a.out))
    print("thresholds chosen on TUNE split:", thr_naive, thr_full)
    for k, v in out["test"].items():
        print(f"{k:55s} hit={v['hit_rate']:.3f} prec={'n/a' if v['hit_precision'] is None else format(v['hit_precision'], '.3f')} rec={v['recall_of_possible_hits']:.3f} wrong={v['wrong_hits']:.1f} llm={v['llm_calls']:.1f}/{n_test}")


def plot(out, d: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, cobalt, plum = "#17212B", "#2350E8", "#7A2E6B"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, col, ls in (("threshold-only", plum, "--"), ("full", cobalt, "-")):
        rows = out["test_sweep"][label]
        x = [r["threshold"] for r in rows]
        ax[0].plot(x, [float("nan") if r["hit_precision"] is None else r["hit_precision"] * 100 for r in rows], ls, color=col, marker="o", ms=4, label=label)
        ax[1].plot(x, [r["recall_of_possible_hits"] * 100 for r in rows], ls, color=col, marker="o", ms=4, label=label)
    ax[0].set(title="Hit precision vs threshold\n(share of cache hits that were the right answer)", xlabel="similarity threshold", ylabel="%", ylim=(0, 102))
    ax[1].set(title="Recall vs threshold\n(share of repeat questions served from cache)", xlabel="similarity threshold", ylabel="%", ylim=(0, 102))
    for a_ in ax:
        a_.axvline(out["threshold_full"], color=cobalt, lw=.9, ls=":")
        a_.axvline(out["threshold_naive"], color=plum, lw=.9, ls=":")
        a_.grid(alpha=.25)
        a_.legend(frameon=False, loc="lower left" if a_ is ax[1] else "lower right")
    fig.suptitle(f"Held-out split, {out['test_intents']} intents, {out['seeds']} shuffles, embedder={out['embedder']}. Dotted lines = thresholds chosen on the tuning split", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(d / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    names = list(out["test"])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    prec = [float("nan") if out["test"][n]["hit_precision"] is None else out["test"][n]["hit_precision"] * 100 for n in names]
    rec = [out["test"][n]["recall_of_possible_hits"] * 100 for n in names]
    wrong = [out["test"][n]["wrong_hits"] for n in names]
    y = range(len(names))
    ax.barh([i - .19 for i in y], prec, .36, color=cobalt, label="hit precision")
    ax.barh([i + .19 for i in y], rec, .36, color=plum, label="recall (repeat questions served from cache)")
    ax.set_yticks(list(y), [n.replace(" @ ", "\n@ ") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 130)
    for i, (pr, rc, w) in enumerate(zip(prec, rec, wrong)):
        label = "no cache hits (no savings)" if pr != pr else f"{w:.1f} wrong answers/run"
        ax.text((0 if pr != pr else max(pr, rc)) + 1.5, i, label, va="center", fontsize=8)
    ax.set_xlabel("%")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Guard ablation on the held-out split", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "guard_ablation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
