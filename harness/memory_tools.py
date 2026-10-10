from __future__ import annotations

import json
import re
import time

from common import call_llm

TOOL_STAGE_EXCLUDED = {"read_video"}

SEMANTIC_TOOLS = {"search_nodes", "search_ocr_text", "enumerate_events", "search_asr_text"}

GRAPH_TOOLS = {"search_by_time", "get_subgraph", "get_summary", "get_super_events",
               "get_macro_events"}

REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "search_nodes": ("query",),
    "search_ocr_text": ("query",),
    "search_asr_text": ("query",),
    "enumerate_events": ("query",),
    "get_subgraph": ("macro_id",),
    "search_by_time": (),
    "get_summary": (),
    "get_super_events": (),
    "get_macro_events": (),
}

EXTRA_PARAMS: dict[str, dict[str, type]] = {
    "search_by_time": {"start_time": str, "end_time": str},
}

_MMSS = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$")


class ToolStageError(RuntimeError):
    pass


def parse_seconds(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str) or not x.strip():
        return None
    m = _MMSS.match(x)
    if m:
        h = float(m.group(1) or 0)
        return h * 3600 + float(m.group(2)) * 60 + float(m.group(3))
    try:
        return float(x.strip())
    except ValueError:
        return None


def _parse_time_range(tr) -> tuple[float, float] | None:
    if not isinstance(tr, (list, tuple)) or len(tr) < 2:
        return None
    s, e = parse_seconds(tr[0]), parse_seconds(tr[1])
    if s is None or e is None or e <= s:
        return None
    return (s, e)


def _normalize_time_window(args: dict) -> tuple[dict, str | None]:
    out: dict = {}
    s = parse_seconds(args.get("start_sec"))
    e = parse_seconds(args.get("end_sec"))
    s_str = args.get("start_time")
    e_str = args.get("end_time")

    if s is None and isinstance(s_str, str):
        s = parse_seconds(s_str)
    if e is None and isinstance(e_str, str):
        e = parse_seconds(e_str)

    if s is None or e is None:
        return {}, ("search_by_time needs a time window: give start_sec/end_sec "
                    "(numbers) or start_time/end_time (MM:SS). "
                    f"got start={args.get('start_sec', args.get('start_time'))!r} "
                    f"end={args.get('end_sec', args.get('end_time'))!r}")
    if s < 0:
        s = 0.0
    if e <= s:
        return {}, f"search_by_time needs end > start, got start={s} end={e}."

    out["start_sec"] = float(s)
    out["end_sec"] = float(e)
    if isinstance(s_str, str) and s_str.strip():
        out["start_time"] = s_str
    if isinstance(e_str, str) and e_str.strip():
        out["end_time"] = e_str
    return out, None


def normalize_args(name: str, args: dict, spec: dict) -> tuple[dict, str | None]:
    if not isinstance(args, dict):
        return {}, f"args must be a JSON object, got {type(args).__name__}."

    params = dict(spec.get("params") or {})
    params.update(EXTRA_PARAMS.get(name, {}))

    if name == "search_by_time":
        return _normalize_time_window(args)

    clean: dict = {}
    for k, typ in params.items():
        if k not in args or args[k] is None:
            continue
        v = args[k]
        try:
            if typ is list:
                clean[k] = list(v) if isinstance(v, (list, tuple)) else [v]
            elif typ is int:
                clean[k] = int(v)
            elif typ is float:
                clean[k] = float(v)
            else:
                clean[k] = str(v)
        except (TypeError, ValueError):
            return {}, f"{name}: argument {k!r}={v!r} is not a valid {typ.__name__}."

    missing = [k for k in REQUIRED_PARAMS.get(name, ()) if k not in clean
               or (isinstance(clean[k], str) and not clean[k].strip())]
    if missing:
        return {}, f"{name}: missing required argument(s) {', '.join(missing)}."

    if name == "get_macro_events" and not clean.get("super_id") and not clean.get("macro_ids"):
        return {}, ("get_macro_events needs either super_id or macro_ids; "
                    "calling it with neither just lists the whole video.")

    unknown = [k for k in args if k not in params]
    if unknown:
        clean["__ignored__"] = unknown
    return clean, None


def learned_tool_specs(qtype: str, g: dict) -> list[dict]:
    mt = (g.get("policy") or {}).get("memory_tools") or {}
    if not mt.get("allow_learned_tools", True):
        return []
    try:
        import tool_forge

        out = []
        for spec in tool_forge.tools_for_type(qtype or ""):
            out.append({
                "name": spec["name"],
                "learned": True,
                "code": spec.get("code") or "",
                "params": spec.get("params") or {},
                "desc": (spec.get("doc") or spec.get("note")
                         or "verified helper tool produced by tool_forge"),
            })
        return out
    except Exception:
        return []


def build_catalog(g: dict, qtype: str, enabled: list[str], specs: dict) -> dict:
    cat: dict[str, dict] = {}
    for name in enabled:
        if name in TOOL_STAGE_EXCLUDED or name not in specs:
            continue
        cat[name] = {**specs[name], "learned": False}
    for spec in learned_tool_specs(qtype, g):
        if spec["name"] not in cat:
            cat[spec["name"]] = spec
    return cat


def render_catalog(cat: dict) -> str:
    lines = []
    for name, spec in cat.items():
        params = dict(spec.get("params") or {})
        params.update(EXTRA_PARAMS.get(name, {}))
        if params:
            args = ", ".join(
                f"{k}: {v.__name__ if isinstance(v, type) else v}" for k, v in params.items()
            )
        else:
            args = "no arguments"
        req = REQUIRED_PARAMS.get(name)
        tail = f" [required: {', '.join(req)}]" if req else ""
        mark = " (learned)" if spec.get("learned") else ""
        lines.append(f"- {name}({args}){mark}{tail}\n    {spec.get('desc', '')}")
    return "\n".join(lines) if lines else "(no memory tools enabled)"


def _scan_json(text: str, marker: str) -> tuple[dict | None, bool]:
    m = re.search(rf"{marker}\s*:?\s*", text)
    if not m:
        return None, False
    frag = text[m.end():]
    if not frag.lstrip().startswith("{"):
        return None, True
    frag = frag[len(frag) - len(frag.lstrip()):]
    depth, in_str, esc, end = 0, False, False, -1
    for i, ch in enumerate(frag):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None, True
    try:
        obj = json.loads(frag[:end])
    except json.JSONDecodeError:
        return None, True
    return (obj if isinstance(obj, dict) else None), True


def parse_stage_reply(text: str) -> dict:
    text = text or ""
    tool_obj, has_tool = _scan_json(text, "TOOL")
    sel_obj, _ = _scan_json(text, "SELECT")
    has_done = re.search(r"\bDONE\b", text) is not None
    done_obj, _ = _scan_json(text, "DONE") if has_done else (None, False)

    select = sel_obj or done_obj

    if has_tool and has_done:
        return {"kind": "error", "why": "output contains both TOOL and DONE; send exactly one",
                "select": select}
    if has_done:
        return {"kind": "done", "select": select}
    if has_tool:
        if tool_obj is None:
            return {"kind": "error", "why": "TOOL: was not followed by a parsable JSON object",
                    "select": select}
        name = tool_obj.get("name") or tool_obj.get("tool")
        if not isinstance(name, str) or not name.strip():
            return {"kind": "error", "why": "TOOL JSON has no usable 'name' field",
                    "select": select}
        args = tool_obj.get("args") or tool_obj.get("arguments") or {}
        return {"kind": "tool", "name": name.strip(),
                "args": args if isinstance(args, dict) else {}, "select": select}
    if select is not None:
        return {"kind": "select", "select": select}
    return {"kind": "error", "why": "output had neither TOOL:, SELECT: nor DONE", "select": None}


_MICRO_ID_KEYS = ("node_id", "event_id", "text_id", "entity_id")


def harvest(data) -> tuple[list[str], list[str], dict[str, tuple[float, float]]]:
    macros: list[str] = []
    nodes: list[str] = []
    spans: dict[str, tuple[float, float]] = {}

    def walk(x):
        if isinstance(x, dict):
            tr = _parse_time_range(x.get("time_range"))
            mid = x.get("macro_id")
            if isinstance(mid, str) and mid.startswith("macro_"):
                if mid not in macros:
                    macros.append(mid)
                if tr and mid not in spans:
                    spans[mid] = tr
            if str(x.get("node_type") or "") != "macro":
                for k in _MICRO_ID_KEYS:
                    v = x.get(k)
                    if isinstance(v, str) and v:
                        if v not in nodes:
                            nodes.append(v)
                        if tr and v not in spans:
                            spans[v] = tr
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return macros, nodes, spans


def _index_metrics(tk) -> dict:
    index = getattr(tk, "index", None)
    getter = getattr(index, "get_thread_metrics", None) or getattr(index, "get_metrics", None)
    try:
        return getter() if callable(getter) else {}
    except Exception:
        return {}


def _backend_status(name: str, before: dict, after: dict) -> dict:
    if name not in SEMANTIC_TOOLS:
        return {"kind": "graph", "degraded": False}
    d = {k: float(after.get(k, 0)) - float(before.get(k, 0))
         for k in ("dense_attempts", "dense_successes", "dense_timeouts",
                   "dense_failures", "bm25_fallbacks")}
    if d["bm25_fallbacks"] > 0:
        return {"kind": "semantic", "degraded": True,
                "why": ("embedding_server unavailable, the index fell back to BM25 "
                        f"(timeouts={int(d['dense_timeouts'])}, "
                        f"failures={int(d['dense_failures'])})"),
                "delta": d}
    if d["dense_attempts"] == 0:
        return {"kind": "semantic", "degraded": False, "why": "no dense attempt recorded",
                "delta": d}
    return {"kind": "semantic", "degraded": False, "delta": d}


def execute(tk, name: str, args: dict, cat: dict, g: dict, obs_limit: int) -> dict:
    rec: dict = {
        "tool": name, "args": args, "status": "ok", "error": None, "obs": "",
        "returned_macro_ids": [], "returned_node_ids": [], "spans": {},
        "backend": {"kind": "unknown", "degraded": False}, "latency_s": 0.0,
        "learned": bool((cat.get(name) or {}).get("learned")),
    }
    spec = cat.get(name)
    if spec is None:
        rec["status"] = "unknown_tool"
        rec["error"] = (f"unknown or disabled tool {name!r}. "
                        f"Available: {', '.join(cat) or '(none)'}")
        rec["obs"] = f"ERROR: {rec['error']}"
        return rec

    clean, err = normalize_args(name, args, spec)
    if err:
        rec["status"] = "bad_args"
        rec["error"] = err
        rec["obs"] = f"ERROR: {err} Nothing was executed -- fix the arguments and retry."
        return rec
    ignored = clean.pop("__ignored__", None)
    rec["clean_args"] = dict(clean)
    if ignored:
        rec["ignored_args"] = ignored

    before = _index_metrics(tk)
    t0 = time.time()
    try:
        if spec.get("learned"):
            import tool_forge

            mt = (g.get("policy") or {}).get("memory_tools") or {}
            raw = tool_forge.run_tool(spec.get("code") or "", name, tk, clean,
                                      timeout=float(mt.get("learned_tool_timeout_sec", 20.0)))
        else:
            fn = getattr(tk, name, None)
            if not callable(fn):
                raise AttributeError(f"toolkit has no method {name!r}")
            raw = fn(**clean)
    except Exception as exc:
        rec["latency_s"] = round(time.time() - t0, 4)
        rec["status"] = "tool_error"
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        rec["obs"] = f"ERROR calling {name}: {rec['error']}"
        rec["backend"] = _backend_status(name, before, _index_metrics(tk))
        return rec
    rec["latency_s"] = round(time.time() - t0, 4)
    rec["backend"] = _backend_status(name, before, _index_metrics(tk))

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        data = raw
    macros, nodes, spans = harvest(data)
    rec["returned_macro_ids"] = macros
    rec["returned_node_ids"] = nodes
    rec["spans"] = spans

    text = raw if isinstance(raw, str) else json.dumps(data, ensure_ascii=False, default=str)
    if not str(text).strip() or (isinstance(data, (list, dict)) and len(data) == 0):
        rec["status"] = "empty"
        rec["obs"] = (f"{name} returned no results. Try different arguments "
                      f"or a different tool.")
        return rec

    if rec["backend"].get("degraded"):
        mt = (g.get("policy") or {}).get("memory_tools") or {}
        mode = str(mt.get("on_embedding_failure", "record_and_continue")).lower()
        if mode == "record_and_reject":
            rec["status"] = "backend_degraded"
            rec["obs"] = (f"ERROR: {name} could not use the semantic index "
                          f"({rec['backend'].get('why')}). Its result is rejected by "
                          f"policy.memory_tools.on_embedding_failure=record_and_reject. "
                          f"Use a graph tool instead (search_by_time / get_macro_events / "
                          f"get_subgraph), which needs no embedding.")
            rec["returned_macro_ids"], rec["returned_node_ids"], rec["spans"] = [], [], {}
            return rec
        rec["status"] = "ok_degraded"

    text = str(text)
    if len(text) > obs_limit:
        text = text[:obs_limit] + f"\n... [truncated at {obs_limit} chars]"
    if rec["status"] == "ok_degraded":
        text = f"[WARNING: semantic index degraded to BM25 for this call]\n{text}"
    rec["obs"] = text
    return rec


def parse_selection(payload, known_spans: dict, allowed: set[str],
                    cap: int) -> tuple[list[str], dict[str, tuple[float, float]], list[str]]:
    ids: list[str] = []
    spans: dict[str, tuple[float, float]] = {}
    rejected: list[str] = []
    if not isinstance(payload, dict):
        return ids, spans, rejected

    raw = payload.get("candidates")
    if raw is None:
        raw = payload.get("select") or payload.get("macro_ids") or payload.get("ids")
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        return ids, spans, rejected

    for item in raw:
        if isinstance(item, str):
            cid, s, e = item, None, None
        elif isinstance(item, dict):
            cid = (item.get("macro_id") or item.get("id") or item.get("node_id")
                   or item.get("event_id"))
            s = parse_seconds(item.get("start_sec", item.get("start_time")))
            e = parse_seconds(item.get("end_sec", item.get("end_time")))
        else:
            rejected.append(f"{item!r}: not an id or object")
            continue
        if not isinstance(cid, str) or not cid.strip():
            rejected.append(f"{item!r}: no id")
            continue
        cid = cid.strip()
        if cid not in allowed:
            rejected.append(f"{cid}: never returned by any tool or by retrieval")
            continue
        if cid in ids:
            continue

        known = known_spans.get(cid)
        win = None
        if s is not None and e is not None and e > s:
            win = (max(0.0, s), e)
            if known:
                lo, hi = known
                a, b = max(win[0], lo), min(win[1], hi)
                if b > a:
                    win = (a, b)
                else:
                    rejected.append(f"{cid}: window {s}-{e} lies outside its own "
                                    f"time range {lo:.1f}-{hi:.1f}, using the node range")
                    win = known
        elif known:
            win = known
        elif s is not None or e is not None:
            rejected.append(f"{cid}: incomplete window (start={s}, end={e})")

        ids.append(cid)
        if win:
            spans[cid] = win
        if len(ids) >= max(1, cap):
            break
    return ids, spans, rejected


def render_history(history: list[dict], window: int) -> str:
    if not history:
        return "(nothing yet -- this is your first tool round)"
    out = []
    for i, h in enumerate(history, 1):
        args = h.get("clean_args", h.get("args")) or {}
        arg_s = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        head = f"[round {i}] {h.get('tool')}({arg_s}) -> {h.get('status')}"
        if i > len(history) - max(1, window):
            out.append(f"{head}\n{h.get('obs', '')}")
        else:
            mids = h.get("returned_macro_ids") or []
            out.append(f"{head} | {len(mids)} macro(s) returned: "
                       f"{', '.join(mids[:6]) or 'none'} [detail dropped]")
    return "\n\n".join(out)


def _summarize(stage: dict) -> str:
    if not stage.get("rounds"):
        return ""
    lines = ["\nMEMORY TOOL STAGE (the solver queried the graph memory itself):"]
    for i, h in enumerate(stage["rounds"], 1):
        args = h.get("clean_args", h.get("args")) or {}
        arg_s = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:4])
        lines.append(f"[{i}] {h.get('tool')}({arg_s}) -> {h.get('status')}")
        if h.get("error"):
            lines.append(f"    error: {h['error']}")
        else:
            body = str(h.get("obs", ""))
            lines.append("    " + body[:600].replace("\n", "\n    "))
    if stage.get("selected"):
        sel = ", ".join(
            f"{cid}"
            + (f"[{stage['selected_spans'][cid][0]:.0f}-{stage['selected_spans'][cid][1]:.0f}s]"
               if cid in stage.get("selected_spans", {}) else "")
            for cid in stage["selected"]
        )
        lines.append(f"SELECTED BY THE SOLVER: {sel}")
        if stage.get("select_why"):
            lines.append(f"  why: {stage['select_why']}")
    else:
        lines.append("SELECTED BY THE SOLVER: (none -- falling back to the initial "
                     "retrieval candidates)")
    return "\n".join(lines)


def stage_enabled(g: dict, mode: str, tk) -> bool:
    if mode != "retrieve" or tk is None:
        return False
    if (g.get("answer") or {}).get("mode") != "pipeline":
        return False
    return bool(((g.get("policy") or {}).get("memory_tools") or {}).get("enabled"))


def run_tool_stage(q: dict, g: dict, cfg: dict, tk, *, enabled: list[str],
                   specs: dict, base_ids: list[str],
                   base_spans: dict | None = None,
                   evidence: str = "") -> dict:
    from pathlib import Path

    from common import VEH_DIR

    mt = (g.get("policy") or {}).get("memory_tools") or {}
    stage: dict = {
        "rounds": [], "selected": [], "selected_spans": {}, "select_why": "",
        "returned_macro_ids": [], "returned_node_ids": [],
        "stop_reason": "", "llm_calls": 0, "tool_calls": 0, "parse_errors": 0,
        "tool_errors": 0, "empty_results": 0, "degraded_calls": 0,
        "rejected_selections": [], "error": None, "evidence_text": "",
        "latency_s": 0.0,
    }
    t_stage = time.time()

    cat = build_catalog(g, q.get("type", ""), enabled, specs)
    if not cat:
        stage["stop_reason"] = "no_tools"
        return stage

    tpl_rel = mt.get("prompt_file") or "prompts/pipeline_tools_v1.txt"
    try:
        tpl = (Path(VEH_DIR) / tpl_rel).read_text(encoding="utf-8")
    except Exception as exc:
        stage["stop_reason"] = "prompt_missing"
        stage["error"] = f"tool-stage prompt unavailable ({tpl_rel}): {str(exc)[:160]}"
        return stage

    max_rounds = max(1, int(mt.get("max_rounds", 3)))
    max_parse_errors = max(1, int(mt.get("max_parse_errors", 2)))
    cap = max(1, int(mt.get("max_selected", 3)))
    obs_limit = int(mt.get("max_obs_chars", 1200))
    obs_window = int(mt.get("obs_window", 2))
    max_record = int(mt.get("max_returned_recorded", 40))

    known_spans: dict[str, tuple[float, float]] = {}
    for k, v in (base_spans or {}).items():
        tr = _parse_time_range(v)
        if tr:
            known_spans[k] = tr
    allowed: set[str] = set(base_ids or [])

    nudge = ""
    while len(stage["rounds"]) < max_rounds:
        left = max_rounds - len(stage["rounds"])
        try:
            prompt = tpl.format(
                tool_list=render_catalog(cat),
                question=str(q.get("question", ""))[:800],
                options="\n".join(str(o) for o in (q.get("options") or [])),
                retrieved=str(evidence)[:1500] or "(nothing retrieved yet)",
                candidates=", ".join(base_ids[:12]) or "(none)",
                observations=render_history(stage["rounds"], obs_window),
                round=len(stage["rounds"]) + 1,
                max_rounds=max_rounds,
                rounds_left=left,
                max_selected=cap,
            ) + nudge
        except KeyError as exc:
            stage["stop_reason"] = "prompt_missing"
            stage["error"] = f"tool-stage prompt has an unknown placeholder {exc}"
            break
        nudge = ""

        if left <= 1 and not stage["selected"]:
            prompt += ("\n\nTHIS IS YOUR LAST ROUND. Do not call another tool -- the "
                       "result would be discarded. Output SELECT or DONE with the "
                       "candidates you want looked at, using ids you have already seen. "
                       "If nothing matches the question, pick the closest segment anyway "
                       "and say why: a rough location still gets frames read, selecting "
                       "nothing does not.\n")

        try:
            text = call_llm(prompt, cfg, int(mt.get("max_tokens", 400)),
                            float(mt.get("temperature", 0.0)), role="pipeline_tools")
            stage["llm_calls"] += 1
        except Exception as exc:
            stage["stop_reason"] = "llm_error"
            stage["error"] = f"tool-stage LLM call failed: {str(exc)[:200]}"
            break

        reply = parse_stage_reply(text)

        if reply.get("select") is not None:
            ids, spans, rej = parse_selection(reply["select"], known_spans, allowed, cap)
            if ids:
                stage["selected"] = ids
                stage["selected_spans"] = spans
                why = reply["select"].get("why") if isinstance(reply["select"], dict) else ""
                stage["select_why"] = str(why or "")[:200]
            stage["rejected_selections"].extend(rej[:6])

        if reply["kind"] == "done":
            stage["stop_reason"] = "done"
            break
        if reply["kind"] == "select":
            continue
        if reply["kind"] == "error":
            stage["parse_errors"] += 1
            tail = str(text or "")[-240:]
            stage.setdefault("parse_error_samples", []).append(
                {"why": reply.get("why"), "reply_tail": tail,
                 "reply_chars": len(str(text or "")),
                 "looks_truncated": not str(text or "").rstrip().endswith(("}", "DONE"))})
            if stage["parse_errors"] >= max_parse_errors:
                stage["stop_reason"] = "parse_errors"
                stage["error"] = f"protocol not followed: {reply.get('why')}"
                break
            nudge = ("\n\nFORMAT ERROR: " + str(reply.get("why")) + "\n"
                     "Send EXACTLY one of:\n"
                     'TOOL: {"name": "...", "args": {...}}\n'
                     'SELECT: {"candidates": [...], "why": "..."}\n'
                     "DONE\n")
            continue

        rec = execute(tk, reply["name"], reply["args"], cat, g, obs_limit)
        stage["rounds"].append(rec)
        stage["tool_calls"] += 1
        if rec["status"] in ("tool_error", "bad_args", "unknown_tool", "backend_degraded"):
            stage["tool_errors"] += 1
        if rec["status"] == "empty":
            stage["empty_results"] += 1
        if rec["backend"].get("degraded"):
            stage["degraded_calls"] += 1

        for mid in rec["returned_macro_ids"]:
            allowed.add(mid)
            if mid not in stage["returned_macro_ids"]:
                stage["returned_macro_ids"].append(mid)
        for nid in rec["returned_node_ids"]:
            allowed.add(nid)
            if nid not in stage["returned_node_ids"]:
                stage["returned_node_ids"].append(nid)
        for k, v in rec["spans"].items():
            known_spans.setdefault(k, v)
    else:
        stage["stop_reason"] = "max_rounds"

    if not stage["stop_reason"]:
        stage["stop_reason"] = "max_rounds"

    if len(stage["returned_macro_ids"]) > max_record:
        stage["returned_macro_truncated"] = len(stage["returned_macro_ids"])
        stage["returned_macro_ids"] = stage["returned_macro_ids"][:max_record]
    if len(stage["returned_node_ids"]) > max_record:
        stage["returned_node_truncated"] = len(stage["returned_node_ids"])
        stage["returned_node_ids"] = stage["returned_node_ids"][:max_record]

    stage["evidence_text"] = _summarize(stage)
    stage["latency_s"] = round(time.time() - t_stage, 4)
    return stage


def merge_findings(base_ids: list[str], base_spans: dict, stage: dict,
                   cap: int = 8) -> tuple[list[str], dict]:
    ids = list(base_ids or [])
    spans = dict(base_spans or {})
    sel = [c for c in (stage.get("selected") or []) if c]
    out: list[str] = []
    for cid in sel + ids:
        if cid not in out:
            out.append(cid)
    for cid, win in (stage.get("selected_spans") or {}).items():
        spans[cid] = tuple(win)
    return out[:max(1, cap)], spans
