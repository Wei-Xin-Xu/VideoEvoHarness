from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ask import TAXONOMY, target_weight
from common import STORE, load_genome, load_questions

def distribution_distance(qs: list[dict]) -> float:
    return distribution_breakdown(qs)["total"]

def _quota_blocked(qtype: str) -> bool:
    spec = TAXONOMY["types"].get(qtype) or {}
    wg = spec.get("weight_gen")
    return wg is not None and float(wg) == 0.0

def distribution_breakdown(qs: list[dict]) -> dict:
    types = sorted(set(TAXONOMY["types"]) | {q["type"] for q in qs})
    total_w = sum(target_weight(t) for t in TAXONOMY["types"]) or 1.0
    n = len(qs)
    c = Counter(q["type"] for q in qs)

    per_type: dict[str, dict] = {}
    reach = block = 0.0
    for t in types:
        share = (c.get(t, 0) / n) if n else 0.0
        tgt = target_weight(t) / total_w
        diff = share - tgt
        blocked = _quota_blocked(t)
        per_type[t] = {"n": c.get(t, 0), "share": round(share, 4),
                       "target": round(tgt, 4), "diff": round(diff, 4),
                       "blocked": blocked}
        if blocked:
            block += abs(diff)
        else:
            reach += abs(diff)
    return {
        "total": ((reach + block) / 2.0) if n else 1.0,
        "reachable": (reach / 2.0) if n else 1.0,
        "floor": block / 2.0,
        "per_type": per_type,
    }

def type_difficulty(solver_by_type: dict[str, object] | None) -> tuple[dict[str, float], str]:
    if not solver_by_type:
        return {}, "none"
    out: dict[str, float] = {}
    used_grounded = False
    fell_back = False
    for t, v in solver_by_type.items():
        if isinstance(v, dict):
            if v.get("acc_grounded") is not None:
                out[t] = float(v["acc_grounded"])
                used_grounded = True
                continue
            fell_back = True
            if v.get("acc") is not None:
                out[t] = float(v["acc"])
        elif v is not None:
            out[t] = float(v)
    if used_grounded:
        return out, "acc_grounded+acc" if fell_back else "acc_grounded"
    return out, "acc"

def zpd_fraction(solver_by_type: dict[str, object], qs: list[dict], lo: float, hi: float) -> float:
    if not qs or not solver_by_type:
        return 0.0
    acc, _metric = type_difficulty(solver_by_type)
    if not acc:
        return 0.0
    hit = sum(1 for q in qs if lo <= acc.get(q["type"], -1.0) <= hi)
    return hit / len(qs)

def score(
    qs_all: list[dict],
    blind_acc: float | None = None,
    oracle_acc: float | None = None,
    solver_by_type: dict[str, object] | None = None,
    g: dict | None = None,
) -> dict:
    g = g or load_genome()
    obj = g["questioner_objective"]
    w = obj["weights"]
    n_opt = g["questioner"]["n_options"]
    chance = 1.0 / max(2, n_opt)
    blind_target = obj.get("blind_target", chance)

    kept = [q for q in qs_all if q.get("verdict") == "keep"]
    reject_rate = 1.0 - (len(kept) / len(qs_all)) if qs_all else 1.0

    if blind_acc is None:
        flags = [q["blind_correct"] for q in qs_all if q.get("blind_correct") is not None]
        blind_acc = sum(flags) / len(flags) if flags else 0.0
    oracle_n = sum(1 for q in qs_all if q.get("oracle_correct") is not None)
    oracle_measured = oracle_acc is not None or oracle_n > 0
    if oracle_acc is None:
        flags = [q["oracle_correct"] for q in qs_all if q.get("oracle_correct") is not None]
        oracle_acc = (sum(flags) / len(flags)) if flags else None

    excess_blind = max(0.0, (blind_acc - chance) / (1.0 - chance))
    excess_reject = max(0.0, reject_rate - chance)

    bd_kept = distribution_breakdown(kept) if kept else {"total": 1.0, "reachable": 1.0,
                                                         "floor": 0.0, "per_type": {}}
    bd_all = distribution_breakdown(qs_all) if qs_all else {"total": 1.0, "reachable": 1.0,
                                                            "floor": 0.0, "per_type": {}}
    dist_kept = bd_kept["reachable"]
    dist_all = bd_all["reachable"]
    lo, hi = obj.get("zpd_range", [0.3, 0.7])
    zpd = zpd_fraction(solver_by_type or {}, kept or qs_all, lo, hi)
    _zpd_acc, zpd_metric = type_difficulty(solver_by_type)

    terms = [
        ("blind", -1.0, w["blind"], excess_blind),
        ("oracle", +1.0, w["oracle"], oracle_acc),
        ("reject", -1.0, w["reject"], excess_reject),
        ("distribution", -1.0, w["distribution"], dist_kept),
        ("zpd", +1.0, w["zpd"], zpd),
    ]
    live = [(n, s, wt, v) for n, s, wt, v in terms if v is not None]
    dropped = sum(wt for _n, _s, wt, v in terms if v is None)
    live_w = sum(wt for _n, _s, wt, _v in live)
    total = 0.0
    breakdown: dict[str, dict] = {}
    for name, s, wt, v in terms:
        measured = v is not None
        share = (wt / live_w) if (measured and live_w > 0) else 0.0
        eff_w = (wt + dropped * share) if measured else 0.0
        contrib = round(s * eff_w * v, 4) if measured else 0.0
        if measured:
            total += contrib
        breakdown[name] = {
            "measured": measured,
            "sign": int(s),
            "weight": wt,
            "effective_weight": round(eff_w, 4),
            "value": round(v, 4) if measured else None,
            "contribution": contrib,
        }
    kept_blind = sum(1 for q in kept if q.get("blind_correct"))
    return {
        "score": round(total, 4),
        "terms": breakdown,
        "terms_dropped": [n for n, d in breakdown.items() if not d["measured"]],
        "weight_redistributed": round(dropped, 4),
        "zpd_metric": zpd_metric,
        "generation": {
            "n_generated": len(qs_all),
            "blind_acc_raw": round(blind_acc, 4),
            "chance": round(chance, 4),
            "excess_blind": round(excess_blind, 4),
            "reject_rate": round(reject_rate, 4),
            "excess_reject": round(excess_reject, 4),
            "yield": round(len(kept) / len(qs_all), 4) if qs_all else 0.0,
            "distribution_distance": round(dist_all, 4),
            "distribution_total": round(bd_all["total"], 4),
            "distribution_floor": round(bd_all["floor"], 4),
        },
        "store": {
            "n_kept": len(kept),
            "blind_solvable_in_store": kept_blind,
            "oracle_acc": round(oracle_acc, 4) if oracle_acc is not None else None,
            "oracle_measured": oracle_measured,
            "oracle_n": oracle_n,
            "distribution_distance": round(dist_kept, 4),
            "distribution_total": round(bd_kept["total"], 4),
            "distribution_floor": round(bd_kept["floor"], 4),
            "distribution_by_type": bd_kept["per_type"],
            "zpd_fraction": round(zpd, 4),
            "zpd_metric": zpd_metric,
        },
        "n_total": len(qs_all),
        "n_kept": len(kept),
        "blind_acc": round(blind_acc, 4),
        "blind_target": round(blind_target, 4),
        "blind_gap": round(abs(blind_acc - blind_target), 4),
        "excess_blind": round(excess_blind, 4),
        "oracle_acc": round(oracle_acc, 4) if oracle_acc is not None else None,
        "reject_rate": round(reject_rate, 4),
        "distribution_distance": round(dist_kept, 4),
        "zpd_fraction": round(zpd, 4),
    }

def per_type_report(qs_all: list[dict], chance: float = 0.25) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in qs_all:
        d = out.setdefault(q["type"], {"n": 0, "blind_correct": 0, "kept": 0})
        d["n"] += 1
        d["blind_correct"] += bool(q.get("blind_correct"))
        d["kept"] += q.get("verdict") == "keep"
    for t, d in out.items():
        d["blind_acc"] = round(d["blind_correct"] / d["n"], 3)
        d["keep_rate"] = round(d["kept"] / d["n"], 3)
        d["excess_blind"] = round(max(0.0, (d["blind_acc"] - chance) / (1.0 - chance)), 3)
        d["gen_per_kept"] = round(1.0 / d["keep_rate"], 2) if d["keep_rate"] else None
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["excess_blind"]))

def bench_report(qs_all: list[dict], key: str, chance: float = 0.25) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in qs_all:
        d = out.setdefault(str(q.get(key) or "unmapped"), {"n": 0, "blind_correct": 0, "kept": 0})
        d["n"] += 1
        d["blind_correct"] += bool(q.get("blind_correct"))
        d["kept"] += q.get("verdict") == "keep"
    for _k, d in out.items():
        d["blind_acc"] = round(d["blind_correct"] / d["n"], 3)
        d["keep_rate"] = round(d["kept"] / d["n"], 3)
        d["excess_blind"] = round(max(0.0, (d["blind_acc"] - chance) / (1.0 - chance)), 3)
        d["gen_per_kept"] = round(1.0 / d["keep_rate"], 2) if d["keep_rate"] else None
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))

def _print_table(rows: dict[str, dict], label: str) -> None:
    for t, d in rows.items():
        flag = "  <- form still leaks" if d["excess_blind"] > 0.30 else ""
        gpk = f"{d['gen_per_kept']:.2f}" if d["gen_per_kept"] else "-"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(STORE / "questions_all_verdict.jsonl"),
                    help="store WITH verdicts, produced by verify.py")
    args = ap.parse_args()

    qs = load_questions(Path(args.questions))
    if not qs:
        raise SystemExit(f"No questions in {args.questions}; run ask.py + verify.py first")
    if not any("verdict" in q for q in qs):
        raise SystemExit("This store has no verdicts; run verify.py first")

    g = load_genome()
    s = score(qs, g=g)

    n_opt = g["questioner"]["n_options"]
    chance = 1.0 / max(2, n_opt)

    by_type = per_type_report(qs, chance)
    by_vmme = bench_report(qs, "videomme_task_type", chance)
    by_cg = bench_report(qs, "cgbench_class", chance)
    _print_table(by_type, "internal taxonomy")
    _print_table(by_vmme, "Video-MME task type")
    _print_table(by_cg, "CG-Bench class")

    out = STORE / "runs" / "qscore.json"
    out.write_text(json.dumps({"summary": s, "by_type": by_type,
                               "by_videomme_task_type": by_vmme,
                               "by_cgbench_class": by_cg},
                              ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
