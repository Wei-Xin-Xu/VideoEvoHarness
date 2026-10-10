from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import solve
from common import (
    QUESTIONS,
    STORE,
    assign_split,
    load_cfg,
    load_genome,
    load_questions,
    save_questions,
)

def verify(questions: list[dict], g: dict, cfg: dict, workers: int = 4) -> list[dict]:
    v = g["verifier"]

    blind = {r["qid"]: r for r in solve.run(questions, g, cfg, "none", workers, quiet=True)}

    need_oracle = [
        q
        for q in questions
        if not blind[q["qid"]]["correct"]
        and not (v["skip_for_structural"] and q.get("source") == "structural")
    ]
    oracle = {}
    if need_oracle:
        oracle = {r["qid"]: r for r in solve.run(need_oracle, g, cfg, "all", workers, quiet=True)}
    else:
        pass

    kept: list[dict] = []
    for q in questions:
        b = blind[q["qid"]]
        q["blind_correct"] = b["correct"]
        o = oracle.get(q["qid"])
        q["oracle_correct"] = o["correct"] if o else None

        if b["correct"]:
            q["verdict"] = "reject"
            q["reject_reason"] = "blind_solvable"
        elif o is not None and not o["correct"]:
            q["verdict"] = "reject"
            q["reject_reason"] = "oracle_unsolvable"
        else:
            q["verdict"] = "keep"
            q["reject_reason"] = None
            q["regret_ready"] = True
            kept.append(q)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--questions", default=None)
    args = ap.parse_args()

    g, cfg = load_genome(), load_cfg()
    qpath = Path(args.questions) if args.questions else QUESTIONS
    qs = load_questions(qpath)
    if args.limit:
        qs = qs[: args.limit]
    if not qs:
        raise SystemExit("Question store is empty; run ask.py first")

    kept = verify(qs, g, cfg, args.workers)

    kept = assign_split(kept, g["split"]["evolve_ratio"], g["split"]["seed"])

    save_questions(kept, STORE / "questions_verified.jsonl")
    save_questions(qs, STORE / "questions_all_verdict.jsonl")

    (STORE / "runs" / "verify_summary.json").write_text(
        json.dumps(
            {
                "n_before": len(qs),
                "n_kept": len(kept),
                "reject_reasons": dict(Counter(q["reject_reason"] for q in qs if q["verdict"] == "reject")),
                "kept_by_type": dict(Counter(q["type"] for q in kept)),
                "rejected_by_type": dict(Counter(q["type"] for q in qs if q["verdict"] == "reject")),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
