"""Fill DIVL replay to 25 unique initial states per task in three LIBERO suites.

Run with the LIBERO Python environment and an already running policy server.
Existing matching episodes count towards the quota; rerunning resumes missing
initial states. Both successful and failed episodes count.
"""

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal")


def existing_episodes(directory, policy_id):
    records = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as e:
            if str(e["policy_id"]) != policy_id:
                continue
            key = (str(e["suite"]), int(e["task_id"]), int(e["init_id"]))
            records.setdefault(key, {"file": str(path), "success": bool(e["success"])})
    return records


def missing_ranges(present, quota):
    """Return contiguous ranges of missing initial states, starting at zero."""
    missing = [i for i in range(quota) if i not in present]
    ranges = []
    for i in missing:
        if ranges and ranges[-1][0] + ranges[-1][1] == i:
            start, count = ranges[-1]
            ranges[-1] = (start, count + 1)
        else:
            ranges.append((i, 1))
    return ranges


def write_status(path, records, quota, status, **details):
    suites = {}
    for suite in SUITES:
        entries = [v for (s, t, i), v in records.items() if s == suite and 0 <= t < 10 and 0 <= i < quota]
        successes = sum(v["success"] for v in entries)
        suites[suite] = {
            "target": 10 * quota,
            "completed": len(entries),
            "successes": successes,
            "failures": len(entries) - successes,
        }
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": datetime.now(UTC).isoformat(),
        "suites": suites,
        **details,
    }
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=25)
    parser.add_argument("--policy-id", default="pi05_libero")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=Path, default=Path("data/divl_libero/episodes"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/divl_libero/collection_run"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.run_dir = args.run_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    records = existing_episodes(args.output, args.policy_id)
    status_file = args.run_dir / "status.json"
    write_status(status_file, records, args.episodes_per_task, "running")
    try:
        for suite in SUITES:
            for task in range(10):
                records = existing_episodes(args.output, args.policy_id)
                present = {i for (s, t, i) in records if s == suite and t == task}
                for start, count in missing_ranges(present, args.episodes_per_task):
                    command = [
                        sys.executable,
                        str(root / "examples/libero/collect_divl.py"),
                        "--host",
                        args.host,
                        "--port",
                        str(args.port),
                        "--suite",
                        suite,
                        "--task-id",
                        str(task),
                        "--start-episode",
                        str(start),
                        "--episodes",
                        str(count),
                        "--policy-id",
                        args.policy_id,
                        "--output",
                        str(args.output),
                    ]
                    log_file = args.run_dir / f"{suite}_task{task:02d}_start{start:02d}.log"
                    print(f"START {suite} task={task} init={start}..{start + count - 1} log={log_file}", flush=True)
                    write_status(
                        status_file,
                        records,
                        args.episodes_per_task,
                        "running",
                        current_suite=suite,
                        current_task=task,
                        current_log=str(log_file),
                    )
                    with log_file.open("a") as log:
                        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                    records = existing_episodes(args.output, args.policy_id)
                    completed = {i for (s, t, i) in records if s == suite and t == task}
                    if not set(range(start, start + count)).issubset(completed):
                        raise RuntimeError(f"Missing output episodes: {suite} task={task}")
                    print(f"DONE {suite} task={task} init={start}..{start + count - 1}", flush=True)
                    write_status(status_file, records, args.episodes_per_task, "running")
        write_status(status_file, records, args.episodes_per_task, "complete")
        print("COMPLETE: all three suite quotas reached", flush=True)
    except Exception as error:
        records = existing_episodes(args.output, args.policy_id)
        write_status(status_file, records, args.episodes_per_task, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
