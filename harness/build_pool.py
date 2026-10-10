from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(os.environ.get("BENCH_DIR") or _ROOT / "benchmarks")
EVO_DIR = Path(os.environ.get("EVO_DIR") or _ROOT / "evolve_sources")
OUT_DIR = Path(__file__).resolve().parent / "store"
VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")

BANDS = [
    ("short", 0.0, 5.0),
    ("medium", 5.0, 15.0),
    ("long", 15.0, 30.0),
    ("very_long", 30.0, 60.0),
    ("extreme", 60.0, 1e9),
]

PROFILES = {
    "balanced": {"short": 0.10, "medium": 0.25, "long": 0.35, "very_long": 0.20, "extreme": 0.10},
    "videomme_long": {"short": 0.0, "medium": 0.10, "long": 0.30, "very_long": 0.50, "extreme": 0.10},
    "cgbench": {"short": 0.05, "medium": 0.20, "long": 0.50, "very_long": 0.20, "extreme": 0.05},
}

def ffprobe_duration(path: str) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None

def scan_videos(root: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for dp, _, fs in os.walk(root):
        if "/.cache/" in dp:
            continue
        for f in fs:
            if f.lower().endswith(VIDEO_EXT):
                found.append((os.path.splitext(f)[0], os.path.join(dp, f)))
    return found

def durations_parallel(items: list[tuple[str, str]], workers: int = 64) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ffprobe_duration, p): vid for vid, p in items}
        for fut in cf.as_completed(futs):
            d = fut.result()
            if d and d > 0:
                out.append((futs[fut], d))
    return out

TEST_BENCHES = [
    "videomme", "videomme_v2", "cgbench", "longvideobench",
    "lvbench", "mlvu_test", "mmvu", "video_holmes", "videommmu",
]

def build_blacklist() -> None:
    ids: set[str] = set()
    per_bench: dict[str, int] = {}
    items: list[tuple[str, str]] = []

    for b in TEST_BENCHES:
        root = BENCH_DIR / b
        if not root.exists():
            continue
        found = scan_videos(root)
        if b == "video_holmes":
            test_json = root / "test_Video-Holmes.json"
            if test_json.exists():
                keep = {str(x.get("video ID")) for x in json.loads(test_json.read_text())}
                found = [(k, p) for k, p in found if k in keep]
        per_bench[b] = len(found)
        ids |= {k for k, _ in found}
        items += found

    lr_ann = BENCH_DIR / "longvideoreason" / "data"
    lr_vid = BENCH_DIR / "longvideoreason_videos"
    if lr_ann.exists() and lr_vid.exists():
        try:
            import glob as _g

            import pandas as pd

            f = _g.glob(str(lr_ann / "test-*.parquet"))
            if f:
                df = pd.read_parquet(f[0])
                keep = {os.path.splitext(os.path.basename(str(v)))[0] for v in df["videos"]}
                found = [(k, p) for k, p in scan_videos(lr_vid) if k in keep]
                per_bench["longvideoreason_test"] = len(found)
                ids |= {k for k, _ in found}
                items += found
        except Exception as exc:
            pass

    for b, n in sorted(per_bench.items(), key=lambda x: -x[1]):
        pass

    pairs = durations_parallel(items)
    durs: dict[str, list[float]] = collections.defaultdict(list)
    for vid, d in pairs:
        durs[vid].append(round(d, 1))
    durs = {k: sorted(set(v)) for k, v in durs.items()}
    multi = sum(1 for v in durs.values() if len(v) > 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bench_video_ids.json").write_text(json.dumps(sorted(ids), indent=0))
    (OUT_DIR / "bench_durations.json").write_text(json.dumps(durs))

def candidate_sources() -> dict[str, Path]:
    src = {
        "mlvu_dev": EVO_DIR / "mlvu_dev",
        "video_holmes_train": EVO_DIR / "video_holmes_train",
        "videomarathon": EVO_DIR / "videomarathon",
        "tvqa_long": EVO_DIR / "tvqa_long",
        "scalelong": EVO_DIR / "scalelong",
        "vript_long": EVO_DIR / "vript_long",
    }
    return {k: v for k, v in src.items() if v.exists()}

def build_pool(n: int, profile: str) -> None:
    bl_path = OUT_DIR / "bench_video_ids.json"
    if not bl_path.exists():
        raise SystemExit("run --blacklist first")
    blacklist = set(json.loads(bl_path.read_text()))
    bench_durs = json.loads((OUT_DIR / "bench_durations.json").read_text())

    holmes_train: set[str] = set()
    for tj in (EVO_DIR / "video_holmes_train" / "train_Video-Holmes.json",
               BENCH_DIR / "video_holmes" / "train_Video-Holmes.json"):
        if tj.exists():
            holmes_train = {str(x.get("video ID")) for x in json.loads(tj.read_text())}
            break

    cands: dict[str, tuple[str, str]] = {}
    for name, root in candidate_sources().items():
        found = scan_videos(root)
        if name == "video_holmes_train":
            found = [(k, p) for k, p in found if k in holmes_train]
        for k, v in found:
            cands.setdefault(k, (v, name))

    dur_pairs = durations_parallel([(k, v[0]) for k, v in cands.items()])
    durs = dict(dur_pairs)

    kept, dropped, cleared = {}, [], 0
    for vid, (path, source) in cands.items():
        d = durs.get(vid)
        if d is None:
            continue
        if vid in blacklist:
            bds = bench_durs.get(vid)
            if isinstance(bds, (int, float)):
                bds = [float(bds)]
            if not bds or any(abs(float(bd) - d) <= 2.0 for bd in bds):
                dropped.append(vid)
                continue
            cleared += 1
        kept[vid] = (path, source, d)

    banded: dict[str, list] = collections.defaultdict(list)
    for vid, (path, source, d) in kept.items():
        m = d / 60.0
        for bname, lo, hi in BANDS:
            if lo <= m < hi:
                banded[bname].append((vid, path, source, d))
                break

    for bname, _, _ in BANDS:
        pass

    shares = PROFILES[profile]
    import random

    rnd = random.Random(42)
    pool = []
    for bname, _, _ in BANDS:
        want = round(n * shares.get(bname, 0.0))
        have = banded[bname]
        rnd.shuffle(have)
        take = have[:want]
        if len(take) < want:
            pass
        pool += [
            {"video_id": v, "path": p, "source": s, "duration_sec": round(d, 1), "band": bname}
            for v, p, s, d in take
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "evolve_pool.json"
    out.write_text(json.dumps(
        {"profile": profile, "n": len(pool),
         "bands": {b: sum(1 for x in pool if x["band"] == b) for b, _, _ in BANDS},
         "sources": dict(collections.Counter(x["source"] for x in pool)),
         "videos": pool}, indent=2))
    tot = sum(x["duration_sec"] for x in pool) / 3600

def main() -> None:
    global BENCH_DIR, EVO_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--blacklist", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--profile", default="balanced", choices=list(PROFILES))
    ap.add_argument("--bench-dir", default=None)
    ap.add_argument("--evo-dir", default=None)
    a = ap.parse_args()
    if a.bench_dir:
        BENCH_DIR = Path(a.bench_dir)
    if a.evo_dir:
        EVO_DIR = Path(a.evo_dir)
    if a.blacklist:
        build_blacklist()
    if a.pool:
        build_pool(a.n, a.profile)
    if not (a.blacklist or a.pool):
        ap.print_help()

if __name__ == "__main__":
    main()
