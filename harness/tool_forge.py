from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

from common import VEH_DIR, STORE, call_llm, load_cfg, load_questions, load_toolkit

_FORGE_ROOT = Path(os.environ.get("VEH_FORGE_ROOT") or VEH_DIR)
TOOLS_DIR = _FORGE_ROOT / "tools_learned"
CANDIDATES_DIR = TOOLS_DIR / "candidates"
VERSIONS_DIR = TOOLS_DIR / "versions"
for _dir in (TOOLS_DIR, CANDIDATES_DIR, VERSIONS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
REGISTRY = TOOLS_DIR / "registry.json"
CURRENT = TOOLS_DIR / "CURRENT"

ALLOWED_BUILTINS = {
    "len", "range", "sorted", "set", "dict", "list", "tuple", "min", "max", "sum", "abs",
    "int", "float", "str", "bool", "enumerate", "zip", "any", "all", "round", "reversed",
    "isinstance", "json",
}
ALLOWED_TK_METHODS = {
    "search_nodes", "get_subgraph", "search_ocr_text", "enumerate_events",
    "search_by_time", "get_summary", "get_super_events", "get_macro_events",
}
SAFE_METHODS = {
    "get", "keys", "values", "items", "setdefault", "update", "copy",
    "append", "extend", "insert", "pop", "remove", "sort", "reverse", "add", "discard",
    "union", "intersection", "difference",
    "split", "rsplit", "splitlines", "strip", "lstrip", "rstrip", "lower", "upper",
    "title", "replace", "startswith", "endswith", "join", "find", "rfind", "format",
    "isdigit", "isalpha", "isnumeric", "zfill",
    "index", "count",
}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                   ast.Lambda, ast.ClassDef, ast.AsyncFunctionDef, ast.Await)

class _StripSafeImports(ast.NodeTransformer):

    def visit_Import(self, node: ast.Import):
        if all(a.name == "json" and a.asname is None for a in node.names):
            return None
        return node

def normalize_code(code: str) -> str:
    try:
        tree = _StripSafeImports().visit(ast.parse(code))
        ast.fix_missing_locations(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and not n.body:
                n.body = [ast.Pass()]
        return ast.unparse(tree)
    except SyntaxError:
        return code

def check_ast(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1 or len(tree.body) != 1:
        return False, "the tool must be exactly one top-level function definition"
    fn = fns[0]
    if not ast.get_docstring(fn):
        return False, "the function needs a docstring; it is shown to the agent as the tool description"

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return False, f"{type(node).__name__} is not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"dunder attribute {node.attr!r} is not allowed"
            base = node.value
            if isinstance(base, ast.Name) and base.id == "tk":
                if node.attr not in ALLOWED_TK_METHODS:
                    return False, f"tk.{node.attr} is not a permitted toolkit method"
            elif isinstance(base, ast.Name) and base.id == "json":
                if node.attr not in {"loads", "dumps"}:
                    return False, f"json.{node.attr} is not permitted"
            elif node.attr not in SAFE_METHODS:
                return False, (
                    f"method {node.attr!r} is not in the safe list; only tk.<method>, "
                    f"json.loads/dumps and plain container/string methods are allowed"
                )
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"dunder name {node.id!r} is not allowed"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            allowed = ALLOWED_BUILTINS | {a.arg for a in fn.args.args} | _local_names(fn)
            if node.func.id not in allowed:
                return False, f"call to {node.func.id!r} is not in the whitelist"
    return True, ""

def _local_names(fn: ast.FunctionDef) -> set[str]:
    out = {fn.name}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.FunctionDef):
            out.add(n.name)
    return out

_LEARNED_FILENAME_PREFIX = "<learned:"

def _safe_builtins() -> dict:
    return {
        k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in ALLOWED_BUILTINS if k != "json"
    }

def run_tool(code: str, fn_name: str, tk, kwargs: dict, timeout: float = 20.0):
    ok, why = check_ast(code)
    if not ok:
        raise ValueError(f"AST check failed: {why}")

    filename = f"{_LEARNED_FILENAME_PREFIX}{fn_name}>"
    env: dict = {"__builtins__": _safe_builtins(), "json": json}
    try:
        exec(compile(code, filename, "exec"), env)
    except BaseException as exc:
        raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:300]}") from None
    fn = env.get(fn_name)
    if not callable(fn):
        raise RuntimeError(f"{fn_name} is not defined by the generated code")

    deadline = time.monotonic() + float(timeout)
    timed_out: list[bool] = []

    def _tracer(frame, event, arg):
        if not frame.f_code.co_filename.startswith(_LEARNED_FILENAME_PREFIX):
            return None
        if time.monotonic() > deadline:
            timed_out.append(True)
            raise TimeoutError(f"{fn_name} exceeded {timeout}s")
        return _tracer

    old_trace = sys.gettrace()
    sys.settrace(_tracer)
    try:
        return fn(tk, **kwargs)
    except TimeoutError:
        raise TimeoutError(f"{fn_name} exceeded {timeout}s") from None
    except BaseException as exc:
        if timed_out:
            raise TimeoutError(f"{fn_name} exceeded {timeout}s") from None
        raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:300]}") from None
    finally:
        sys.settrace(old_trace)

def verify_tool(spec: dict, tk, timeout: float = 20.0) -> tuple[bool, str]:
    cases = spec.get("tests") or []
    if not cases:
        return False, "no self-verification cases supplied"
    for i, case in enumerate(cases, 1):
        kwargs = case.get("kwargs") or {}
        try:
            got = run_tool(spec["code"], spec["name"], tk, kwargs, timeout)
        except Exception as exc:
            return False, f"case {i} raised: {str(exc)[:160]}"
        exp = case.get("expect")
        op = case.get("op", "eq")
        if op == "eq" and got != exp:
            return False, f"case {i}: expected {exp!r}, got {got!r}"
        if op == "ge" and not (isinstance(got, (int, float)) and got >= exp):
            return False, f"case {i}: expected >= {exp!r}, got {got!r}"
        if op == "nonempty" and not got:
            return False, f"case {i}: expected a non-empty result, got {got!r}"
        if op == "type" and type(got).__name__ != exp:
            return False, f"case {i}: expected type {exp}, got {type(got).__name__}"
    return True, f"{len(cases)}/{len(cases)} self-verification cases passed"

TYPE_CONTRACTS: dict[str, dict] = {
    "counting": {
        "signature": "count_entities_in_segment(tk, start_sec: float, end_sec: float, entity_type: str)",
        "returns": "int -- how many entity ENTRIES of that entity_type the ONE matching segment holds",
        "note": (
            "Match EXACTLY ONE macro: the one whose own time_range equals [start_sec, end_sec] "
            "(allow a second or two of rounding). tk.search_by_time returns every macro OVERLAPPING "
            "the window, so counting all of them gives a number that is too high -- pick the single "
            "best match, then count inside its subgraph only. "
            "Count ENTRIES, do NOT deduplicate by name or entity_id: the ground truth is "
            "len([e for e in subgraph['entities'] if e['entity_type'].upper() == entity_type.upper()]). "
            "Comparison of entity_type must be case-insensitive."
        ),
    },
}
TYPE_CONTRACTS["count_entity"] = TYPE_CONTRACTS["counting"]

_COUNT_RE = re.compile(r"(?:from\s+)?(\d+):(\d+)\s*(?:-|to)\s*(\d+):(\d+)")
_TYPE_RE = re.compile(r"Event type:\s*(\w+)|distinct\s+(\w+)\s+entit|entity type\s+(\w+)|(PERSON|OBJECT|LOCATION|GROUP)",
                      re.I)

def build_cases_for_type(qtype: str, questions: list[dict]) -> list[dict]:
    cases: list[dict] = []
    if qtype in ("count_entity", "counting"):
        for q in questions:
            if q.get("type") != qtype:
                continue
            m, t = _COUNT_RE.search(q["question"]), _TYPE_RE.search(q["question"])
            if not (m and t):
                continue
            etype = next((x for x in t.groups() if x), "").upper()
            if etype not in {"PERSON", "OBJECT", "LOCATION", "GROUP"}:
                continue
            gold = None
            for opt in q.get("options", []):
                if opt.strip().startswith(q["answer"] + "."):
                    gold = opt.split(".", 1)[1].strip()
            if gold is None or not gold.isdigit():
                continue
            cases.append({
                "kwargs": {
                    "start_sec": int(m.group(1)) * 60 + int(m.group(2)),
                    "end_sec": int(m.group(3)) * 60 + int(m.group(4)),
                    "entity_type": etype,
                },
                "expect": int(gold),
                "op": "eq",
                "macro_id": (q.get("anchor") or {}).get("macro_id"),
                "qid": q.get("qid"),
            })
    return cases

def cross_check(spec: dict, tk, cases: list[dict], min_pass: float = 0.75) -> tuple[bool, str]:
    if not cases:
        return True, "no ground-truth cases available for this type (self-verified only)"
    try:
        sig_args = _param_names(spec["code"])
    except Exception:
        sig_args = []
    want = list((cases[0].get("kwargs") or {}).keys())
    if sig_args and sorted(sig_args) != sorted(want):
        return False, (f"signature mismatch: tool takes {sig_args} but the contract requires "
                       f"{want}. Use the exact signature given in the contract.")
    passed, details = 0, []
    for c in cases:
        kwargs = dict(c["kwargs"])
        try:
            got = run_tool(spec["code"], spec["name"], tk, kwargs)
        except Exception as exc:
            details.append(f"{c.get('qid')}: raised {str(exc)[:60]}")
            continue
        if str(got) == str(c["expect"]):
            passed += 1
        else:
            details.append(f"{c.get('qid')}: got {got}, expected {c['expect']}")
    rate = passed / len(cases)
    msg = f"{passed}/{len(cases)} ground-truth cases ({rate:.0%})"
    if details:
        msg += " | " + "; ".join(details[:4])
    return rate >= min_pass, msg

def _param_names(code: str) -> list[str]:
    fn = [n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)][0]
    return [a.arg for a in fn.args.args if a.arg != "tk"]

def _active_tools_dir() -> Path:
    if CURRENT.exists():
        version = CURRENT.read_text(encoding="utf-8").strip()
        active = VERSIONS_DIR / version
        if active.is_dir() and (active / "registry.json").exists():
            return active
    return TOOLS_DIR

def load_registry() -> dict:
    registry = _active_tools_dir() / "registry.json"
    return json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}

def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def save_registry(reg: dict) -> None:
    _atomic_text(REGISTRY, json.dumps(reg, ensure_ascii=False, indent=2))

def register(spec: dict, note: str) -> None:
    reg = load_registry()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    version = f"{stamp}_{time.time_ns() % 1000000:06d}_{spec['name']}"
    candidate = CANDIDATES_DIR / version
    committed = VERSIONS_DIR / version
    candidate.mkdir(parents=True, exist_ok=False)
    active = _active_tools_dir()
    for old_tool in active.glob("*.py"):
        (candidate / old_tool.name).write_text(old_tool.read_text(encoding="utf-8"), encoding="utf-8")
    (candidate / f"{spec['name']}.py").write_text(spec["code"], encoding="utf-8")
    reg[spec["name"]] = {
        "name": spec["name"],
        "signature": spec.get("signature", ""),
        "desc": spec.get("desc", ""),
        "applies_to": spec.get("applies_to") or [],
        "tests": spec.get("tests") or [],
        "verified": note,
        "version": version,
        "wins": 0,
        "trials": 0,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }
    (candidate / "registry.json").write_text(
        json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    candidate.replace(committed)
    _atomic_text(CURRENT, version + "\n")
    save_registry(reg)
    _atomic_text(TOOLS_DIR / f"{spec['name']}.py", spec["code"])

PENDING_TOOL: dict | None = None

def tools_for_type(qtype: str) -> list[dict]:
    out = []
    pend = PENDING_TOOL
    if pend and qtype in (pend.get("applies_to") or []) and pend.get("code"):
        out.append({**pend, "pending": True})
    active = _active_tools_dir()
    for spec in load_registry().values():
        if qtype not in (spec.get("applies_to") or []):
            continue
        if pend and spec.get("name") == pend.get("name"):
            continue
        path = active / f"{spec['name']}.py"
        if path.exists():
            out.append({**spec, "code": path.read_text(encoding="utf-8")})
    return out

FORGE_PROMPT = """You are writing a Python helper tool for an agent that answers questions about
long videos by querying a hierarchical graph memory. The agent keeps getting
`{qtype}` questions wrong because it tries to do the work by reading text instead of
computing. Write a tool that computes the answer exactly.

## What the tool receives
The first parameter is always `tk`, the memory toolkit. Only these methods exist:
{tk_methods}

Each returns a JSON STRING. Parse with json.loads. Shapes you can rely on:
- get_subgraph(macro_id) -> {{"macro_id","label","time_range","description",
    "entities":[{{"entity_id","name","entity_type","attributes","description"}}],
    "events":[{{"event_id","time_range","subject","action","description"}}],
    "ocr_texts":[{{"text_id","text","time_range"}}], "key_relations":[...]}}
- search_by_time(start_sec, end_sec) -> [{{"macro_id","label","time_range",...}}]
- get_macro_events() -> [{{"macro_id","label","time_range",...}}]
entity_type is one of PERSON / OBJECT / LOCATION / GROUP.

CRITICAL -- time_range is a STRING, never numbers. At every level it looks like
"00:06:24-00:09:12" (HH:MM:SS-HH:MM:SS). Comparing it to a number silently matches
nothing and your tool will return 0 for everything. Parse it first, for example:

    tr = macro.get("time_range") or ""
    parts = tr.split("-")
    if len(parts) == 2:
        a = parts[0].split(":")
        secs = int(a[0]) * 3600 + int(a[1]) * 60 + int(a[2])

Note that int() and split() are available, but there is no datetime module.

## Required signature (contract -- do not deviate)
{contract}

## Hard constraints
- Exactly ONE top-level function. `json` is already available; an `import json` line is
  tolerated but any other import is rejected.
- No attribute access except tk.<method> and json.loads / json.dumps.
- Allowed builtins only: {builtins}
- The function needs a docstring: it becomes the tool description the agent reads.
- Return a plain value (number, string, list or dict), never an object.
- Be defensive: missing keys and empty lists must not raise.

## Failing examples of this question type
{examples}

## Feedback on your previous attempt
{feedback}

## Output
Return ONLY one JSON object:
{{"name": "snake_case_name",
  "signature": "name(tk, arg: type, ...)",
  "desc": "one line, what it computes and when to call it",
  "applies_to": ["{qtype}"],
  "code": "def name(tk, ...):\\n    \\"\\"\\"docstring\\"\\"\\"\\n    ...",
  "tests": [{{"kwargs": {{...}}, "op": "eq|ge|nonempty|type", "expect": ...}}]}}

The tests are executed against the REAL memory of video {video_key} and all must pass,
so choose arguments that genuinely exist there. Prefer `ge` or `nonempty` over `eq` when
you cannot know the exact value. At least two test cases.
"""

def forge(
    qtype: str,
    video_key: str,
    cfg: dict,
    examples: str = "",
    model: str | None = None,
    attempts: int = 3,
    questions: list[dict] | None = None,
    commit: bool = True,
    gate5_mode: str = "full",
    gate5_min_pass: float = 0.75,
) -> dict | None:
    tk = load_toolkit(video_key)
    cases = build_cases_for_type(qtype, questions or [])
    if cases:
        pass
    else:
        pass

    feedback = ""
    for attempt in range(1, attempts + 1):
        con = TYPE_CONTRACTS.get(qtype)
        contract_txt = (
            f"signature: {con['signature']}\nreturns: {con['returns']}\nIMPORTANT: {con['note']}"
            if con else "(no fixed contract for this type; choose a sensible signature)"
        )
        prompt = FORGE_PROMPT.format(
            qtype=qtype,
            contract=contract_txt,
            tk_methods="\n".join(f"- tk.{m}(...)" for m in sorted(ALLOWED_TK_METHODS)),
            builtins=", ".join(sorted(ALLOWED_BUILTINS)),
            examples=examples or "(none supplied)",
            video_key=video_key,
            feedback=feedback or "(first attempt, no feedback yet)",
        )
        try:
            text = call_llm(prompt, cfg, max_tokens=2000, temperature=0.3,
                            role="tool_forge", model=model)
        except Exception as exc:
            return None

        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            feedback = "Your previous reply contained no JSON object. Return only the JSON object."
            continue
        try:
            spec = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            feedback = f"Your previous JSON failed to parse: {exc}. Return strictly valid JSON."
            continue
        if not spec.get("name") or not spec.get("code"):
            feedback = "Your previous reply was missing 'name' or 'code'."
            continue

        spec["code"] = normalize_code(textwrap.dedent(spec["code"]))
        spec.setdefault("applies_to", [qtype])

        ok, why = check_ast(spec["code"])
        if not ok:
            feedback = f"Your previous tool was rejected by the sandbox: {why}. Fix it."
            continue

        ok, why = verify_tool(spec, tk)
        if not ok:
            feedback = f"Your own assertions failed: {why}. Fix the tool or the assertions."
            continue

        ok, why = cross_check(spec, tk, cases, min_pass=gate5_min_pass) \
            if gate5_mode != "bypass" else (True, f"skipped (gate5_mode={gate5_mode})")
        enforced = gate5_mode == "full"
        if not ok and enforced:
            feedback = (
                f"Your tool passed its own assertions but DISAGREED with the "
                f"program-computed answers to real questions: {why}\n"
                f"Those answers are ground truth, derived by code from the memory, not by "
                f"a model. Work out what your tool counts that the question does not. "
                f"Hint: a question about one segment refers to exactly one macro event, "
                f"while tk.search_by_time returns every macro OVERLAPPING the window."
            )
            continue
        spec["_gate5_note"] = why
        spec["_gate5_passed"] = bool(ok)
        spec["_gate5_enforced"] = enforced
        spec["_smoke_tested"] = True
        note = f"self: {verify_tool(spec, tk)[1]} | ground truth: {why}"
        if not enforced:
            note += f" | gate5 NOT enforced (mode={gate5_mode})"
        if commit:
            register(spec, note)
        else:
            pass
        return spec

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", metavar="QTYPE", help="forge a tool for this question type")
    ap.add_argument("--video", default="0RxMZBLeqRI", help="video whose memory the tests run against")
    ap.add_argument("--examples", default="", help="failing examples to show the model")
    ap.add_argument("--model", default=None, help="ABLATION ONLY: forge with a different model")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-all", action="store_true", help="re-run every registered tool's tests")
    ap.add_argument("--attempts", type=int, default=3, help="retries with failure fed back in")
    ap.add_argument("--questions", default=str(STORE / "questions_all_verdict.jsonl"),
                    help="question store supplying gate-5 ground truth")
    args = ap.parse_args()

    if args.list:
        reg = load_registry()
        if not reg:
            return
        for name, r in reg.items():
            pass
        return

    if args.verify_all:
        reg = load_registry()
        tk = load_toolkit(args.video)
        active = _active_tools_dir()
        for name, r in reg.items():
            code = (active / f"{name}.py").read_text(encoding="utf-8")
            ok, why = verify_tool({**r, "code": code}, tk)
        return

    if args.forge:
        cfg = load_cfg()
        cfg["QWEN_MM_CHAT_TIMEOUT"] = "120"
        qs = load_questions(Path(args.questions)) if Path(args.questions).exists() else []
        forge(args.forge, args.video, cfg, args.examples, args.model,
              attempts=args.attempts, questions=qs)
        return

    ap.print_help()

if __name__ == "__main__":
    main()
