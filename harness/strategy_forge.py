from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import solve
import tool_forge as tf
from common import VEH_DIR, active_memory_dir, call_llm, load_cfg, load_genome

STRATEGIES_DIR = Path(os.environ.get("VEH_FORGE_ROOT") or VEH_DIR) / "strategies"
CANDIDATES_DIR = STRATEGIES_DIR / "candidates"
VERSIONS_DIR = STRATEGIES_DIR / "versions"
CURRENT = STRATEGIES_DIR / "CURRENT"
REGISTRY = STRATEGIES_DIR / "registry.json"

FN_NAME = "retrieve_evidence"
SIGNATURE = f"{FN_NAME}(tk, question, topk_hint, max_chars, max_macros)"

DEFAULT_TIMEOUT = 25.0

MIN_TYPE_SAMPLES = 6

CHURN_MAX = 1.25

_GATE5_WORKERS_DEFAULT = 1

def _gate5_workers() -> int:
    try:
        n = int(os.environ.get("VEH_GATE5_WORKERS", _GATE5_WORKERS_DEFAULT))
    except (TypeError, ValueError):
        return _GATE5_WORKERS_DEFAULT
    return max(1, n)

EXTRA_BUILTINS = {"Exception", "ValueError", "KeyError", "TypeError", "IndexError",
                  "AttributeError", "ZeroDivisionError"}

EXTRA_SAFE_METHODS = {
    "isupper", "islower", "istitle", "isspace", "isalnum",
    "capitalize", "casefold", "swapcase",
    "partition", "rpartition", "removeprefix", "removesuffix",
}

_SANDBOX_LOCK = threading.RLock()
_SANDBOX_DEPTH = 0
_SANDBOX_SAVED: dict = {}

def _relaxed_sandbox():
    import ast
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with _SANDBOX_LOCK:
            global _SANDBOX_DEPTH
            first = _SANDBOX_DEPTH == 0
            if first:
                _SANDBOX_SAVED["nodes"] = tf.FORBIDDEN_NODES
                _SANDBOX_SAVED["builtins"] = tf.ALLOWED_BUILTINS
                _SANDBOX_SAVED["methods"] = tf.SAFE_METHODS
                tf.FORBIDDEN_NODES = tuple(
                    n for n in _SANDBOX_SAVED["nodes"] if n is not ast.Lambda)
                tf.ALLOWED_BUILTINS = _SANDBOX_SAVED["builtins"] | EXTRA_BUILTINS
                tf.SAFE_METHODS = _SANDBOX_SAVED["methods"] | EXTRA_SAFE_METHODS
            _SANDBOX_DEPTH += 1
            try:
                yield
            finally:
                _SANDBOX_DEPTH -= 1
                if _SANDBOX_DEPTH == 0:
                    tf.FORBIDDEN_NODES = _SANDBOX_SAVED["nodes"]
                    tf.ALLOWED_BUILTINS = _SANDBOX_SAVED["builtins"]
                    tf.SAFE_METHODS = _SANDBOX_SAVED["methods"]

    return _ctx()

def _active_dir() -> Path:
    if CURRENT.exists():
        version = CURRENT.read_text(encoding="utf-8").strip()
        active = VERSIONS_DIR / version
        if active.is_dir() and (active / "registry.json").exists():
            return active
    return STRATEGIES_DIR

def load_registry() -> dict:
    p = _active_dir() / "registry.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

def strategy_for_type(qtype: str | None) -> dict | None:
    if not qtype:
        return None
    reg = load_registry()
    for spec in reg.values():
        if not spec.get("verified"):
            continue
        if qtype in (spec.get("harmful_types") or []):
            continue
        if qtype in (spec.get("applies_to") or []):
            return spec
    return None

def strategy_code(spec: dict) -> str | None:
    path = _active_dir() / f"{spec['name']}.py"
    if not path.exists():
        path = STRATEGIES_DIR / f"{spec['name']}.py"
    return path.read_text(encoding="utf-8") if path.exists() else None

def check_signature(code: str) -> tuple[bool, str]:
    import ast

    try:
        fns = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    except SyntaxError as exc:
        return False, f"cannot parse the code: {exc}"
    if not fns:
        return False, "no function definition found"
    target = next((f for f in fns if f.name == FN_NAME), None)
    if target is None:
        return False, f"the function must be named {FN_NAME}, found {[f.name for f in fns]}"
    names = [a.arg for a in target.args.args]
    want = ["tk", "question", "topk_hint", "max_chars", "max_macros"]
    if names != want:
        return False, f"parameters must be exactly {want}, found {names}"
    return True, "signature is correct"

def check_ast_for_strategy(code: str) -> tuple[bool, str]:
    with _relaxed_sandbox():
        return tf.check_ast(code)

def check_result_shape(out) -> tuple[bool, str]:
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        return False, f"must return (evidence text, segment list), returned {type(out).__name__}"
    text, ids = out
    if not isinstance(text, str):
        return False, f"the first return value must be a string, found {type(text).__name__}"
    if not isinstance(ids, (list, tuple)) or any(not isinstance(x, str) for x in ids):
        return False, "the second return value must be a list of segment id strings"
    return True, "return shape is correct"

def run_strategy(code: str, tk, question: str, g: dict,
                 timeout: float = DEFAULT_TIMEOUT) -> tuple[str, list[str]]:
    r, e = g["retrieval"], g["evidence"]
    with _relaxed_sandbox():
        out = tf.run_tool(code, FN_NAME, tk, {
            "question": question,
            "topk_hint": int(r["topk_macro"]),
            "max_chars": int(e["max_chars"]),
            "max_macros": max(1, int(r["topk_macro"]) * 3),
        }, timeout=timeout)
    ok, why = check_result_shape(out)
    if not ok:
        raise ValueError(why)
    text, ids = out
    return text[:int(e["max_chars"])], list(ids)[:max(1, int(r["topk_macro"]) * 3)]

def cross_check(spec: dict, questions: list[dict], g: dict, cfg: dict,
                churn_max: float | None = None) -> tuple[bool, str, dict]:
    from copy import deepcopy

    g_on = deepcopy(g)
    g_on.setdefault("retrieval", {})["use_learned_strategy"] = True
    g_off = deepcopy(g)
    g_off.setdefault("retrieval", {})["use_learned_strategy"] = False

    solve.PENDING_STRATEGY = spec
    try:
        return _cross_check_inner(spec, questions, g, g_on, g_off, cfg,
                                  churn_max=churn_max)
    finally:
        solve.PENDING_STRATEGY = None

def _cross_check_inner(spec: dict, questions: list[dict], g: dict,
                       g_on: dict, g_off: dict, cfg: dict,
                       churn_max: float | None = None) -> tuple[bool, str, dict]:
    stats = {"n": 0, "strategy_correct": 0, "baseline_correct": 0,
             "strategy_hit": 0, "baseline_hit": 0,
             "strategy_macros": 0, "baseline_macros": 0, "strategy_errors": 0,
             "identical_evidence": 0, "skipped_by_cosine_gate": 0}
    per_type: dict[str, dict[str, int]] = {}

    tks: dict[str, object] = {}
    for vk in {q["video_key"] for q in questions}:
        tks[vk] = solve.load_toolkit(vk)

    abort = threading.Event()
    err_lock = threading.Lock()
    errors: list[str] = []

    def _one(q: dict) -> dict:
        if abort.is_set():
            return {"kind": "skipped"}
        tk = tks[q["video_key"]]
        try:
            run_strategy(spec["code"], tk, q["question"], g)
        except Exception as exc:
            with err_lock:
                errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                if len(errors) >= 3:
                    abort.set()
            return {"kind": "error"}
        probe_on: dict = {}
        ev_on, _ = solve.build_evidence(q, g_on, "retrieve", tk, probe=probe_on)
        ev_off, _ = solve.build_evidence(q, g_off, "retrieve", tk)
        rec_on = solve.solve_one(q, g_on, cfg, "retrieve", tk)
        rec_off = solve.solve_one(q, g_off, cfg, "retrieve", tk)
        return {"kind": "ok", "q": q, "identical": ev_on == ev_off,
                "top_cosine": probe_on.get("top_cosine"),
                "rec_on": rec_on, "rec_off": rec_off}

    def _tally(r: dict) -> None:
        if r["kind"] == "error":
            stats["strategy_errors"] += 1
            return
        if r["kind"] != "ok":
            return
        q = r["q"]
        if r["identical"]:
            stats["identical_evidence"] += 1
            ceiling = (g_on.get("retrieval") or {}).get("strategy_max_cosine")
            top = r["top_cosine"]
            if ceiling is not None and top is not None and top >= float(ceiling):
                stats["skipped_by_cosine_gate"] += 1
        rec_on, rec_off = r["rec_on"], r["rec_off"]
        s_ok = bool(rec_on.get("correct"))
        b_ok = bool(rec_off.get("correct"))
        stats["n"] += 1
        stats["strategy_correct"] += int(s_ok)
        stats["baseline_correct"] += int(b_ok)
        stats["strategy_hit"] += int(bool(rec_on.get("gold_macro_hit")))
        stats["baseline_hit"] += int(bool(rec_off.get("gold_macro_hit")))
        stats["strategy_macros"] += len(rec_on.get("located") or [])
        stats["baseline_macros"] += len(rec_off.get("located") or [])
        t = q.get("type") or "unknown"
        d = per_type.setdefault(t, {"n": 0, "fixed": 0, "broken": 0})
        d["n"] += 1
        d["fixed"] += int(s_ok and not b_ok)
        d["broken"] += int(b_ok and not s_ok)

    def _progress(i: int, total: int, t0: float) -> None:
        if i % 10 and i != total:
            return
        el = time.time() - t0
        eta = el / i * (total - i) if i else 0.0

    gw = _gate5_workers()
    total = len(questions)
    t0 = time.time()
    if gw > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(gw, total)) as pool:
            futs = [pool.submit(_one, q) for q in questions]
            for i, fut in enumerate(as_completed(futs), 1):
                _tally(fut.result())
                _progress(i, total, t0)
    else:
        for i, q in enumerate(questions, 1):
            _tally(_one(q))
            _progress(i, total, t0)

    if stats["strategy_errors"] >= 3:
        return False, f"the strategy fails repeatedly on real questions: {errors[-1] if errors else ''}", stats

    stats["per_type"] = per_type

    if stats["n"] == 0:
        return False, "not a single question completed", stats
    n = stats["n"]
    if stats["identical_evidence"] == n:
        gated = stats["skipped_by_cosine_gate"]
        ceiling = (g_on.get("retrieval") or {}).get("strategy_max_cosine")
        if gated == n:
            return False, (
                f"default retrieval has top cosine >= strategy_max_cosine on all {n} questions"
                f"({ceiling}), so the gate skipped the strategy by design and both sides saw identical evidence."
                f"This gate never exercised the strategy: use questions whose top cosine is below {ceiling},"
                f"or raise that gate temporarily. The integration path itself is fine"), stats
        return False, (f"strategy and baseline produced identical evidence on all {n} questions"
                       f"({gated} of which were skipped by the cosine gate) --"
                       f"for the rest this means the strategy was never actually invoked, so the check is void"), stats
    s_acc, b_acc = stats["strategy_correct"] / n, stats["baseline_correct"] / n
    s_hit, b_hit = stats["strategy_hit"] / n, stats["baseline_hit"] / n
    s_m, b_m = stats["strategy_macros"] / n, stats["baseline_macros"] / n
    stats.update({"strategy_acc": round(s_acc, 4), "baseline_acc": round(b_acc, 4),
                  "strategy_hit_rate": round(s_hit, 4), "baseline_hit_rate": round(b_hit, 4),
                  "strategy_macros_per_q": round(s_m, 2), "baseline_macros_per_q": round(b_m, 2)})
    note = (f"{n} real questions: acc {b_acc:.3f} -> {s_acc:.3f}, "
            f"hit {b_hit:.3f} -> {s_hit:.3f}, segments per question {b_m:.1f} -> {s_m:.1f}")

    if s_acc < b_acc - 1e-9:
        return False, note + "  [rejected] accuracy below the default pipeline", stats

    churn_ratio = s_m / b_m if b_m > 0 else 1.0
    cmax = CHURN_MAX if churn_max is None else churn_max
    acc_gained = s_acc > b_acc + 1e-9
    if s_hit > b_hit + 1e-9 and churn_ratio > cmax and not acc_gained:
        return False, (note + f"  [rejected] hit rate rose but the segment count rose {churn_ratio:.2f}x,"
                              f"so the hit rate was bought with volume; accuracy {b_acc:.3f} -> {s_acc:.3f} "
                              f"did not follow, meaning retrieval did not become more precise"), stats

    if abs(s_acc - b_acc) < 1e-9 and churn_ratio > cmax:
        return False, (note + f"  [rejected] accuracy unchanged while the segment count rose {churn_ratio:.2f}x,"
                              f"so the extra segments produced no benefit"), stats

    good, harmful, thin = [], [], []
    for t, d in sorted(per_type.items()):
        if d["n"] < MIN_TYPE_SAMPLES:
            thin.append(f"{t}({d['n']} items)")
        elif d["fixed"] > d["broken"]:
            good.append(t)
        elif d["broken"] > d["fixed"]:
            harmful.append(t)
    detail = ", ".join(f"{t}(+{d['fixed']}/-{d['broken']}/{d['n']} items)"
                       for t, d in sorted(per_type.items()))
    note += f"\n    by question type: {detail}"
    if thin:
        note += f"\n    not judged, too few samples (need >= {MIN_TYPE_SAMPLES}): {', '.join(thin)}"
    if not good:
        return False, (note + f"\n    [rejected] no question type shows a net gain on at least {MIN_TYPE_SAMPLES}"
                              f" {MIN_TYPE_SAMPLES} samples, so there is no basis for enabling it"), stats
    stats["applies_to"] = good
    stats["harmful_types"] = harmful
    stats["undecided_types"] = thin
    note += f"\n    applies to: {good}"
    if harmful:
        note += f"  excluded: {harmful}"
    return True, note, stats

def _default_strategy_source() -> str:
    return """
# Default pipeline (this is what a candidate strategy replaces):
ranked = []
for entry in genome.retrieval.entries:        # ["nodes", "macro", ...]
    hits = tk.search_nodes(question, top_k=topk*multiplier, node_types=...)
    for item in hits["results"]:
        mid = item["parent_macro"]["macro_id"]
        keep the highest item["score"] seen for each mid
ranked = ranked[:topk]                        # plain truncation
evidence = ""
for mid in ranked:                            # take the whole subgraph per segment
    sg = tk.get_subgraph(mid)
    Take rows from events / entities / ocr_texts in turn until max_chars is exhausted
return evidence, ranked
""".strip()

def build_prompt(diagnosis: dict, for_type: str | None, base_stats: dict | None) -> str:
    return f"""You are writing the retrieval strategy for a long-video question-answering
system. The memory is a hierarchical graph; answering is done from text evidence only,
never by re-watching frames.

## Write exactly one function

def {SIGNATURE}:
    \"\"\"One-line description of what this strategy does.\"\"\"
    ...
    return evidence_text, located_macro_ids

Return values are NOT optional:
- `evidence_text`: the text that goes into the answering prompt, at most `max_chars`.
- `located_macro_ids`: every macro segment you decided is relevant, at most `max_macros`.
  This list is scored against the true segment, so returning everything to inflate the
  hit rate is detected and rejected.

## Available toolkit methods (nothing else exists)

- `tk.search_nodes(query, top_k=N, node_types=[...])` -> JSON with `results`, each item
  has `node_id`, `node_type` (Event/Entity/OnScreenText/macro), `label`, `description`,
  `time_range`, `score` (semantic similarity, higher is better), `parent_macro`.
- `tk.get_subgraph(macro_id)` -> JSON with `macro_id`, `label`, `description`,
  `time_range`, `events`, `entities`, `ocr_texts`, `key_relations`.
- `tk.search_ocr_text(query, top_k=N)` -> on-screen text hits.
- `tk.enumerate_events(query, min_cosine=F, max_results=N)` -> events above a similarity
  floor.
- `tk.search_by_time(start_sec, end_sec)` -> everything in a time window.
- `tk.get_summary()` / `tk.get_super_events()` / `tk.get_macro_events()` -> coarse levels.

Every method returns a JSON string; parse with `json.loads`. `json` is already available.

IMPORTANT about return shapes: a parsed result may be a dict with a `results` key OR a
bare list, depending on the method. Guard for both, e.g.
`data = json.loads(raw); items = data.get("results", []) if isinstance(data, dict) else (data or [])`.
Calling `.get()` on a list is a real failure mode that has already happened.

## Hard constraints

- No `import` of anything. No `open`, `eval`, `exec`, no dunder attributes.
- Builtins limited to: {", ".join(sorted(tf.ALLOWED_BUILTINS | EXTRA_BUILTINS))}
- `try` / `except Exception` IS available, and you should use it around each toolkit
  call so that one malformed payload skips that segment instead of failing the question.
- Container and string methods are available (`get`, `items`, `append`, `sort`, `split`,
  `lower`, `join`, ...). `lambda` IS allowed, so `items.sort(key=lambda x: -x["score"])`
  is fine.
- Exactly ONE top-level function, with a docstring. No helper functions at module level;
  put everything inside `{FN_NAME}`.
- Keep it fast: this runs once per question and is killed after {DEFAULT_TIMEOUT:.0f}s.
  Do not call `get_subgraph` on every segment of the video.
- Never return empty evidence when any candidate was found: an empty string means the
  question is answered with no evidence at all, which scores near chance.

## What the current default does

{_default_strategy_source()}

## Failure attribution on the current default

{json.dumps(diagnosis.get("diagnosis", {}), ensure_ascii=False)}
per question type: {json.dumps(diagnosis.get("by_type", {}), ensure_ascii=False)}

- retrieval_miss: retrieval never returned the segment holding the answer
- evidence_truncated: the fact was in memory but never reached the prompt
- memory_missing_or_reasoning: not in memory, or a reasoning failure
- guess_no_anchor: answered right but the cited evidence does not support it

## Directions that do not work -- do not repeat them

- Only adding more text (extra search entries, a larger `max_chars`, pulling in
  neighbouring segments) does not change which segments are retrieved. It makes the
  prompt longer and tends to cost accuracy.
- Simply raising top-k raises the segment hit rate, but the extra segments are usually
  noise: answers flip in both directions and the net effect is hard to separate from
  run-to-run variation.
- Feeding the whole memory scores below feeding only the segment that holds the answer.
  Irrelevant context actively costs accuracy.

So: more evidence is not the answer, and neither is simply less. What is untried is
*choosing better* -- deciding per question which segments deserve the budget.

## How you will be judged (read this before writing code)

Your strategy is run on real questions side by side with the default pipeline, and it is
REJECTED unless all of the following hold:

1. Accuracy must not drop below the default.
2. If the segment hit rate goes up, the number of segments you return must NOT grow by
   more than 25%. Returning more candidates trivially raises the hit rate without making
   retrieval any better, and such attempts are rejected.
3. If accuracy is unchanged, the segment count must not grow by more than 25% either:
   extra segments that buy nothing are pure cost.
4. At least one question type must come out net positive. Results are broken down per
   type, and your strategy will only be switched on for the types where it actually helps.

About point 4: a strategy that helps one type and hurts another is the normal case, not
the exception. A small aggregate gain can hide a large loss on one type, which is why
the breakdown decides where a strategy is enabled. So you do NOT need to be good at
everything -- declare nothing in `applies_to` and let the measurement decide. Being
clearly good at one or two types is more useful than being mediocre everywhere.

In other words: a higher hit rate only counts if you got there with a comparable number
of segments, and it only matters if accuracy follows. Optimise for picking the RIGHT few
segments, not for covering more of them.
{f"## Target question type{chr(10)}Optimise for `{for_type}` questions specifically." if for_type else ""}
{f"## Current baseline{chr(10)}{json.dumps(base_stats, ensure_ascii=False)}" if base_stats else ""}

## Output format

Return ONLY a JSON object, no prose, no markdown fence:
{{"name": "snake_case_name",
  "desc": "one line",
  "applies_to": [],
  "code": "def {FN_NAME}(tk, question, topk_hint, max_chars, max_macros):\\n    \\"\\"\\"...\\"\\"\\"\\n    ...",
  "tests": [{{"question": "a question to try", "expect_nonempty": true}}]}}

`applies_to` lists question types this is specialised for, or [] for a general strategy.
"""

def parse_spec(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        spec = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(spec, dict) or "code" not in spec:
        return None
    spec["code"] = tf.normalize_code(spec["code"])
    spec.setdefault("name", "strategy")
    spec.setdefault("applies_to", [])
    spec["signature"] = SIGNATURE
    return spec

def register(spec: dict, note: str, stats: dict) -> None:
    reg = load_registry()
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}_{spec['name']}"
    candidate = CANDIDATES_DIR / version
    candidate.mkdir(parents=True, exist_ok=False)
    active = _active_dir()
    for old in active.glob("*.py"):
        (candidate / old.name).write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    (candidate / f"{spec['name']}.py").write_text(spec["code"], encoding="utf-8")
    reg[spec["name"]] = {
        "name": spec["name"], "signature": SIGNATURE, "desc": spec.get("desc", ""),
        "applies_to": stats.get("applies_to") or [],
        "model_claimed_applies_to": spec.get("applies_to") or [],
        "harmful_types": stats.get("harmful_types") or [],
        "verified": True,
        "verified_note": note,
        "cross_check": stats, "version": version,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    (candidate / "registry.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate.replace(VERSIONS_DIR / version)
    tf._atomic_text(CURRENT, version + "\n")

def _probe_workers(default: int = 6) -> int:
    try:
        n = int(os.environ.get("VEH_GATE5_WORKERS", default))
    except (TypeError, ValueError):
        return default
    return max(1, n)

def pick_cases(questions: list[dict], g: dict, cfg: dict, for_type: str | None,
               n_cases: int, workers: int | None = None, min_per_type: int = 0) -> list[dict]:
    pool = [q for q in questions if not for_type or q.get("type") == for_type]
    if not pool:
        return []

    if min_per_type > 0 and not for_type:
        by_type: dict[str, list[dict]] = {}
        for q in pool:
            by_type.setdefault(q.get("type") or "unknown", []).append(q)
        probe: list[dict] = []
        for qs in by_type.values():
            probe += qs[:min_per_type * 3]
        target = f"{min_per_type} per question type ({len(by_type)} types)"
    else:
        by_type = None
        probe = pool[:min(len(pool), max(n_cases * 3, 18))]
        target = f"{n_cases} items"

    nw = _probe_workers() if workers is None else max(1, int(workers))
    t0 = time.time()
    recs = solve.run(probe, g, cfg, "retrieve", nw, quiet=True)
    ok_by_qid = {r.get("qid"): bool(r.get("correct")) for r in recs}

    def split_pick(cands: list[dict], want: int) -> list[dict]:
        wrong = [q for q in cands if not ok_by_qid.get(q["qid"], False)]
        right = [q for q in cands if ok_by_qid.get(q["qid"], False)]
        half = max(1, want // 2)
        picked = wrong[:half] + right[:want - min(half, len(wrong))]
        if len(picked) < want:
            seen = {q["qid"] for q in picked}
            picked += [q for q in cands if q["qid"] not in seen][:want - len(picked)]
        return picked

    if by_type is not None:
        cases: list[dict] = []
        for t, qs in sorted(by_type.items()):
            avail = [q for q in probe if (q.get("type") or "unknown") == t]
            got = split_pick(avail, min_per_type)
            cases += got
            n_wrong = sum(1 for q in got if not ok_by_qid.get(q["qid"], False))
            flag = ""
            if len(got) < min_per_type:
                flag = f"  [too few, the pool only holds {len(avail)}]"
            elif n_wrong == 0 or n_wrong == len(got):
                flag = "  [one-sided, weak discrimination]"
    else:
        cases = split_pick(probe, n_cases)

    n_wrong = sum(1 for q in cases if not ok_by_qid.get(q["qid"], False))
    if n_wrong == 0:
        pass
    return cases

def _smoke_case(questions: list[dict], for_type: str | None) -> list[dict]:
    pools = []
    if for_type:
        pools.append([q for q in questions if q.get("type") == for_type])
    pools.append(questions)
    for pool in pools:
        for q in sorted(pool, key=lambda x: str(x.get("qid"))):
            vk = q.get("video_key")
            if not vk:
                continue
            try:
                if Path(active_memory_dir(vk)).is_dir():
                    return [q]
            except Exception:
                continue
    return []

def forge(cfg: dict, g: dict, questions: list[dict], diagnosis: dict,
          for_type: str | None, n_cases: int, attempts: int,
          min_per_type: int = 0, commit: bool = True,
          gate5_mode: str = "full", churn_max: float | None = None,
          ) -> tuple[bool, str, dict | None]:
    if gate5_mode == "bypass":
        cases = _smoke_case(questions, for_type)
        if cases:
            pass
        else:
            pass
    else:
        cases = pick_cases(questions, g, cfg, for_type, n_cases, min_per_type=min_per_type)
    if not cases and gate5_mode == "full":
        why = f"no usable real questions (for_type={for_type}). Gate 5 cannot run; aborting."
        return False, why, None
    if cases and gate5_mode != "bypass":
        pass
    elif not cases:
        pass

    feedback = ""
    for attempt in range(1, attempts + 1):
        prompt = build_prompt(diagnosis, for_type, None)
        if feedback:
            prompt += f"\n## why the previous attempt failed; please fix it\n{feedback}\n"
        try:
            text = call_llm(prompt, cfg, max_tokens=2600, temperature=0.4, role="evolver")
        except Exception as exc:
            why = f"model call failed: {str(exc)[:200]}"
            return False, why, None
        spec = parse_spec(text)
        if not spec:
            feedback = "the reply is not a valid JSON object, or the code field is missing. Return JSON only, with no explanatory text."
            continue

        ok, why = check_signature(spec["code"])
        if not ok:
            feedback = f"wrong signature: {why}"
            continue
        ok, why = check_ast_for_strategy(spec["code"])
        if not ok:
            feedback = f"sandbox check failed: {why}"
            continue

        probe = cases[0] if cases else None
        if probe is None:
            smoke_tested = False
        else:
            try:
                tk = solve.load_toolkit(probe["video_key"])
                text_out, ids = run_strategy(spec["code"], tk, probe["question"], g)
            except Exception as exc:
                feedback = f"execution against real memory failed: {type(exc).__name__}: {str(exc)[:240]}"
                continue
            if not text_out.strip():
                feedback = "the returned evidence is an empty string, which is the same as no evidence at answering time"
                continue
            smoke_tested = True

        enforced = gate5_mode == "full"
        if gate5_mode == "bypass" or not cases:
            ok, note, stats = True, f"gate 5 not executed (mode={gate5_mode}, {len(cases)} real questions)", {}
        else:
            ok, note, stats = cross_check(spec, cases, g, cfg, churn_max=churn_max)
        if not ok and enforced:
            feedback = (f"cross-validation on real questions failed: {note}\n"
                        f"Note: accuracy must not fall below the default pipeline, and any hit-rate gain must not come from returning more segments.")
            continue

        spec["_cross_note"] = note
        spec["_cross_stats"] = stats
        spec["_gate5_passed"] = bool(ok)
        spec["_gate5_enforced"] = enforced
        spec["_smoke_tested"] = smoke_tested
        if not enforced:
            note += f"  | gate 5 not enforced (mode={gate5_mode}); this is not a validated strategy"
        if commit:
            register(spec, note, stats)
        return True, note, spec

    return False, feedback or f"all {attempts} attempts failed", None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", action="store_true", help="generate a new strategy")
    ap.add_argument("--for-type", default=None, help="optimise for one question type")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--cases", type=int, default=8,
                    help="how many real questions gate 5 uses (applies when no per-type quota is set)")
    ap.add_argument("--min-per-type", type=int, default=MIN_TYPE_SAMPLES,
                    help=f"minimum questions per type (default {MIN_TYPE_SAMPLES}). "
                         f"This sets the reliability of the per-type verdict; the total is this times the number of types. "
                         f"Set 0 to fall back to a total given by --cases")
    ap.add_argument("--attempts", type=int, default=3, help="how many times to retry after a failed generation")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt only, do not call the model")
    args = ap.parse_args()

    if args.list:
        reg = load_registry()
        if not reg:
            return
        for name, spec in reg.items():
            pass
        return

    diag_path = Path(os.environ.get(
        "VEH_STORE_DIR", VEH_DIR / "store")) / "runs" / f"diagnose_v{load_genome()['version']}_evolve.json"
    diagnosis: dict = {}
    if diag_path.exists():
        diagnosis = json.loads(diag_path.read_text(encoding="utf-8"))
    else:
        pass

    if args.dry_run:
        return

    if not args.forge:
        ap.error("one of --forge, --list or --dry-run is required")

    cfg = load_cfg()
    g = load_genome()
    qs_path = args.questions or Path(
        os.environ.get("VEH_STORE_DIR", VEH_DIR / "store")) / "questions_verified.jsonl"
    if not qs_path.exists():
        raise SystemExit(f"question pool {qs_path} not found; gate 5 cannot validate without real questions, aborting.")
    questions = [json.loads(l) for l in qs_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    ok, why, spec = forge(cfg, g, questions, diagnosis, args.for_type, args.cases,
                          args.attempts, min_per_type=args.min_per_type)
    if ok:
        pass
    else:
        raise SystemExit(f"generation failed: {why}")

if __name__ == "__main__":
    main()
