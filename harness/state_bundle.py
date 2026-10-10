from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VEH_DIR = Path(__file__).resolve().parent
ROOT = VEH_DIR.parent

BUNDLE_VERSION = 1
MANIFEST_NAME = "manifest.json"
ADAPTATION_MODES = ("frozen", "evolve")

SOLVER_PROTOCOL = "graph_memory_pipeline_v1"

MUTATING_CAPABILITIES = (
    "genome_commit",
    "genome_snapshot",
    "memory_commit",
    "memory_restore",
    "tool_register",
    "strategy_commit",
    "prompt_commit",
    "questioner_commit",
    "candidate_generate",
)

SKIP_NAMES = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".log"}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def _keep(path: Path) -> bool:
    if any(part in SKIP_NAMES for part in path.parts):
        return False
    return path.suffix.lower() not in SKIP_SUFFIXES

def hash_tree(root: Path) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if not _keep(rel):
            continue
        out[rel.as_posix()] = sha256_file(path)
    return out

def tree_hash(root: Path) -> str:
    return sha256_json(hash_tree(root))

def git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            capture_output=True, timeout=15, check=False,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None

def source_code_hash() -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in sorted(VEH_DIR.glob("*.py")):
        files[f"harness/{path.name}"] = sha256_file(path)
    for name in ("solve.py",):
        if (VEH_DIR / name).exists():
            files[f"harness/{name}"] = sha256_file(VEH_DIR / name)
    for path in sorted(ROOT.glob("*.py")):
        files[path.name] = sha256_file(path)
    return {"files": files, "sha256": sha256_json(files)}

def resolve_memory_dir(memory_root: Path, video_key: str) -> tuple[Path, str]:
    base = Path(memory_root) / video_key
    current = base / "CURRENT"
    if current.exists():
        version = current.read_text(encoding="utf-8").strip()
        candidate = (base / "versions" / version).resolve()
        if candidate.parent == (base / "versions").resolve() and (candidate / "graph_memory.json").exists():
            return candidate, version
    return base, "base"

def pin_memory(
    memory_root: Path,
    video_keys: Iterable[str] | None = None,
    hash_embeddings: bool = True,
) -> dict:
    memory_root = Path(memory_root).expanduser().resolve()
    if video_keys is None:
        keys = sorted(
            d.name for d in memory_root.iterdir()
            if d.is_dir() and ((d / "graph_memory.json").exists() or (d / "CURRENT").exists())
        ) if memory_root.is_dir() else []
    else:
        keys = sorted(set(video_keys))

    videos: dict[str, dict] = {}
    for key in keys:
        directory, version = resolve_memory_dir(memory_root, key)
        graph = directory / "graph_memory.json"
        index = directory / "embeddings.npz"
        record: dict[str, Any] = {
            "version": version,
            "dir": str(directory),
            "graph_exists": graph.exists(),
            "index_exists": index.exists(),
        }
        if graph.exists():
            record["graph_sha256"] = sha256_file(graph)
            record["graph_bytes"] = graph.stat().st_size
            try:
                raw = json.loads(graph.read_text(encoding="utf-8"))
                record["macro_count"] = len(raw.get("macro_events") or [])
            except Exception as exc:
                record["graph_error"] = str(exc)[:300]
        if index.exists():
            record["index_bytes"] = index.stat().st_size
            if hash_embeddings:
                record["index_sha256"] = sha256_file(index)
        videos[key] = record

    payload = {
        "created_at": utc_now(),
        "memory_root": str(memory_root),
        "video_count": len(videos),
        "ready_count": sum(bool(v.get("graph_exists") and v.get("index_exists")) for v in videos.values()),
        "embeddings_hashed": bool(hash_embeddings),
        "videos": videos,
    }
    payload["sha256"] = sha256_json(payload["videos"])
    return payload

def _copy_tree(src: Path, dest: Path) -> None:
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        if not _keep(rel):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

def snapshot(
    dest: Path,
    *,
    genome_path: Path | None = None,
    prompts_dir: Path | None = None,
    tools_dir: Path | None = None,
    strategies_dir: Path | None = None,
    memory_root: Path | None = None,
    video_keys: Iterable[str] | None = None,
    questions: dict[str, Path] | None = None,
    config_file: Path | None = None,
    model: str | None = None,
    embed_model: str | None = None,
    runtime: dict | None = None,
    note: str = "",
    hash_embeddings: bool = True,
    overwrite: bool = False,
) -> dict:
    dest = Path(dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise FileExistsError(f"bundle directory is not empty: {dest} (pass --overwrite to replace)")
    if dest.exists() and overwrite:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    genome_path = Path(genome_path or (VEH_DIR / "genome.json")).expanduser().resolve()
    prompts_dir = Path(prompts_dir or (VEH_DIR / "prompts")).expanduser().resolve()
    tools_dir = Path(tools_dir or (VEH_DIR / "tools_learned")).expanduser().resolve()
    strategies_dir = Path(strategies_dir or (VEH_DIR / "strategies")).expanduser().resolve()

    genome = json.loads(genome_path.read_text(encoding="utf-8"))
    write_json(dest / "genome.json", genome)

    _copy_tree(prompts_dir, dest / "prompts")
    _copy_tree(tools_dir, dest / "tools")

    (dest / "strategies").mkdir(parents=True, exist_ok=True)
    if strategies_dir.is_dir():
        _copy_tree(strategies_dir, dest / "strategies")
    if not (dest / "strategies" / "strategies.json").exists():
        write_json(dest / "strategies" / "strategies.json", {
            "version": 0,
            "note": "placeholder pin: pipeline-strategy DSL not yet enabled; both arms load this file",
            "strategies": {},
        })

    question_records: dict[str, dict] = {}
    for name, path in (questions or {}).items():
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"question manifest not found: {src}")
        target = dest / "questions" / f"{name}{src.suffix or '.jsonl'}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        rows = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
        question_records[name] = {
            "source": str(src),
            "bundle_path": target.relative_to(dest).as_posix(),
            "sha256": sha256_file(target),
            "rows": rows,
        }

    memory_payload = None
    if memory_root:
        memory_payload = pin_memory(memory_root, video_keys, hash_embeddings=hash_embeddings)
        write_json(dest / "memory_versions.json", memory_payload)

    code = source_code_hash()
    write_json(dest / "source_code.json", code)

    prompt_files = hash_tree(dest / "prompts")
    tool_files = hash_tree(dest / "tools")
    strategy_files = hash_tree(dest / "strategies")

    immutable = {
        "bundle_version": BUNDLE_VERSION,
        "solver_protocol": SOLVER_PROTOCOL,
        "genome_version": genome.get("version"),
        "genome_sha256": sha256_json(genome),
        "answer_mode": (genome.get("answer") or {}).get("mode"),
        "pipeline_tool_stage": bool(((genome.get("policy") or {})
                                    .get("memory_tools") or {}).get("enabled")),
        "require_replay_on_retrieve": bool(((genome.get("policy") or {})
                                           .get("read_video") or {})
                                          .get("require_replay_on_retrieve", True)),
        "prompt_tree_sha256": sha256_json(prompt_files),
        "tool_registry_sha256": sha256_json(tool_files),
        "strategy_tree_sha256": sha256_json(strategy_files),
        "memory_root": str(Path(memory_root).expanduser().resolve()) if memory_root else None,
        "memory_manifest_sha256": (memory_payload or {}).get("sha256"),
        "memory_video_count": (memory_payload or {}).get("video_count", 0),
        "memory_ready_count": (memory_payload or {}).get("ready_count", 0),
        "question_manifests": question_records,
        "source_code_sha256": code["sha256"],
        "model": model,
        "embed_model": embed_model,
        "runtime": runtime or {},
    }
    snapshot_id = sha256_json(immutable)
    manifest = {
        "created_at": utc_now(),
        "note": note,
        "git_commit": git_commit(),
        "config_file": str(Path(config_file).expanduser().resolve()) if config_file else None,
        "snapshot_id": snapshot_id,
        "protocol_fingerprint": snapshot_id,
        "prompt_files": prompt_files,
        "tool_files": tool_files,
        "strategy_files": strategy_files,
        **immutable,
    }
    write_json(dest / MANIFEST_NAME, manifest)
    (dest / "README.txt").write_text(
        "Immutable VideoEvoHarness State Bundle.\n"
        "Do not edit any file in this directory. The frozen arm resolves all adaptive\n"
        "state from here, and `state_bundle.py verify` recomputes every digest.\n",
        encoding="utf-8",
    )
    return manifest

def load_manifest(bundle: Path) -> dict:
    path = Path(bundle).expanduser().resolve() / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"not a state bundle (missing {MANIFEST_NAME}): {bundle}")
    return json.loads(path.read_text(encoding="utf-8"))

def verify(bundle: Path, skip_memory: bool = False, hash_embeddings: bool = True) -> dict:
    bundle = Path(bundle).expanduser().resolve()
    manifest = load_manifest(bundle)
    mismatches: list[dict] = []

    def check(name: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append({"artifact": name, "expected": expected, "actual": actual})

    genome = json.loads((bundle / "genome.json").read_text(encoding="utf-8"))
    check("genome", manifest.get("genome_sha256"), sha256_json(genome))
    check("prompts", manifest.get("prompt_tree_sha256"), sha256_json(hash_tree(bundle / "prompts")))
    check("tools", manifest.get("tool_registry_sha256"), sha256_json(hash_tree(bundle / "tools")))
    check("strategies", manifest.get("strategy_tree_sha256"), sha256_json(hash_tree(bundle / "strategies")))
    for name, record in (manifest.get("question_manifests") or {}).items():
        path = bundle / record["bundle_path"]
        check(f"questions:{name}", record["sha256"], sha256_file(path) if path.exists() else None)
    check("source_code", manifest.get("source_code_sha256"), source_code_hash()["sha256"])

    memory_report: dict[str, Any] = {"checked": False}
    pinned = bundle / "memory_versions.json"
    if not skip_memory and pinned.exists():
        recorded = json.loads(pinned.read_text(encoding="utf-8"))
        recomputed = pin_memory(
            recorded["memory_root"], list((recorded.get("videos") or {}).keys()),
            hash_embeddings=hash_embeddings and recorded.get("embeddings_hashed", True),
        )
        drifted = []
        for key, want in (recorded.get("videos") or {}).items():
            got = (recomputed.get("videos") or {}).get(key) or {}
            for field in ("version", "graph_sha256", "index_sha256", "index_bytes"):
                if field in want and want.get(field) != got.get(field):
                    drifted.append({"video_key": key, "field": field,
                                    "expected": want.get(field), "actual": got.get(field)})
        memory_report = {
            "checked": True,
            "memory_root": recorded["memory_root"],
            "video_count": recorded.get("video_count"),
            "drifted": drifted,
        }
        if drifted:
            mismatches.append({"artifact": "memory", "drifted": len(drifted)})

    return {
        "checked_at": utc_now(),
        "bundle": str(bundle),
        "snapshot_id": manifest.get("snapshot_id"),
        "state_hash": sha256_json({
            "genome": sha256_json(genome),
            "prompts": sha256_json(hash_tree(bundle / "prompts")),
            "tools": sha256_json(hash_tree(bundle / "tools")),
            "strategies": sha256_json(hash_tree(bundle / "strategies")),
            "memory": (json.loads(pinned.read_text(encoding="utf-8")).get("sha256")
                       if pinned.exists() else None),
        }),
        "state_unchanged": not mismatches,
        "mismatches": mismatches,
        "memory": memory_report,
    }

def env_for(
    bundle: Path,
    adaptation_mode: str = "frozen",
    arm: str | None = None,
    checkpoint: int = 0,
    memory_root: Path | None = None,
    memory_pins: Path | None = None,
) -> dict[str, str]:
    if adaptation_mode not in ADAPTATION_MODES:
        raise ValueError(f"adaptation_mode must be one of {ADAPTATION_MODES}")
    bundle = Path(bundle).expanduser().resolve()
    manifest = load_manifest(bundle)
    env = {
        "VEH_STATE_BUNDLE": str(bundle),
        "VEH_ADAPTATION_MODE": adaptation_mode,
        "VEH_ARM": arm or ("frozen" if adaptation_mode == "frozen" else "evolving"),
        "VEH_CHECKPOINT": str(int(checkpoint)),
        "VEH_GENOME_PATH": str(bundle / "genome.json"),
        "VEH_PROMPT_ROOT": str(bundle),
        "VEH_TOOLS_DIR": str(bundle / "tools"),
        "VEH_STRATEGY_DIR": str(bundle / "strategies"),
    }
    root = memory_root or manifest.get("memory_root")
    if root:
        env["VEH_MEMORY_ROOT"] = str(Path(root).expanduser().resolve())
    pins = Path(memory_pins) if memory_pins else (bundle / "memory_versions.json")
    if Path(pins).exists():
        env["VEH_MEMORY_VERSION_MAP"] = str(Path(pins).expanduser().resolve())
    return env

def fingerprint(parts: dict) -> str:
    return sha256_json(parts)

def _kv(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"expected name=path, got: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description="VideoEvoHarness immutable State Bundle")
    ap.add_argument("command", choices=["snapshot", "pin-memory", "verify", "show", "env"])
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--genome", default=None)
    ap.add_argument("--prompts-dir", default=None)
    ap.add_argument("--tools-dir", default=None)
    ap.add_argument("--strategies-dir", default=None)
    ap.add_argument("--memory-root", default=None)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--questions", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--config-file", default=str(ROOT / "qwen-mm-config"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--adaptation-mode", choices=list(ADAPTATION_MODES), default="frozen")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--checkpoint", type=int, default=0)
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--no-embedding-hash", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.command == "snapshot":
        if not args.dest:
            raise SystemExit("snapshot requires --dest")
        manifest = snapshot(
            Path(args.dest),
            genome_path=args.genome,
            prompts_dir=args.prompts_dir,
            tools_dir=args.tools_dir,
            strategies_dir=args.strategies_dir,
            memory_root=args.memory_root,
            video_keys=args.videos,
            questions=_kv(args.questions),
            config_file=args.config_file,
            model=args.model,
            embed_model=args.embed_model,
            note=args.note,
            hash_embeddings=not args.no_embedding_hash,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "pin-memory":
        if not args.memory_root:
            raise SystemExit("pin-memory requires --memory-root")
        payload = pin_memory(Path(args.memory_root), args.videos,
                             hash_embeddings=not args.no_embedding_hash)
        if args.out:
            write_json(Path(args.out), payload)
        return 0

    if not args.bundle:
        raise SystemExit(f"{args.command} requires --bundle")

    if args.command == "verify":
        report = verify(Path(args.bundle), args.skip_memory, not args.no_embedding_hash)
        return 0 if report["state_unchanged"] else 3

    if args.command == "show":
        return 0

    env = env_for(Path(args.bundle), args.adaptation_mode, args.arm, args.checkpoint,
                  args.memory_root)
    for key, value in env.items():
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
