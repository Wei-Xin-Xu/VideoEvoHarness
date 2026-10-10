from __future__ import annotations

import collections
import json
from pathlib import Path

EVOLVE_SPLIT = "evolve"


def build(questions: list[dict], n_shards: int, seed: int = 42) -> dict:
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, received {n_shards}")

    ev = [q for q in questions if q.get("split") == EVOLVE_SPLIT]
    if not ev:
        raise ValueError(
            "the evolve pool is empty, so it cannot be sharded. Check the split field in questions_verified.jsonl")

    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for q in ev:
        by_video[q.get("video_key")].append(q)

    order = sorted(by_video.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    bins: list[dict] = [{"videos": [], "n_questions": 0} for _ in range(n_shards)]
    for vk, qs in order:
        b = min(bins, key=lambda x: (x["n_questions"], len(x["videos"])))
        b["videos"].append(vk)
        b["n_questions"] += len(qs)

    for b in bins:
        rc: dict[str, int] = {}
        for vk in b["videos"]:
            for q in by_video[vk]:
                r = q.get("route") or "unrouted"
                rc[r] = rc.get(r, 0) + 1
        b["by_route"] = rc
        b["n_videos"] = len(b["videos"])
        b["videos"].sort()

    return {
        "n_shards": n_shards,
        "seed": seed,
        "split": EVOLVE_SPLIT,
        "n_videos_total": len(by_video),
        "n_questions_total": len(ev),
        "shards": bins,
    }


def save(plan: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def shard_index(plan: dict, round_index: int) -> int:
    n = len(plan["shards"]) or 1
    return round_index % n


def videos_of(plan: dict, round_index: int) -> list[str]:
    return list(plan["shards"][shard_index(plan, round_index)]["videos"])


def select(questions: list[dict], plan: dict, round_index: int) -> list[dict]:
    vids = set(videos_of(plan, round_index))
    return [q for q in questions
            if q.get("split") == EVOLVE_SPLIT and q.get("video_key") in vids]


def describe(plan: dict, round_index: int | None = None) -> str:
    lines = [f"sharding plan: {plan['n_shards']} shards / "
             f"{plan['n_videos_total']} videos / {plan['n_questions_total']} evolve questions"]
    for i, b in enumerate(plan["shards"]):
        mark = " <- this round" if round_index is not None and i == shard_index(plan, round_index) else ""
        lines.append(f"  shard {i}: {b['n_videos']:>3} videos  {b['n_questions']:>4}  "
                     f"{json.dumps(b['by_route'], ensure_ascii=False)}{mark}")
    return "\n".join(lines)
