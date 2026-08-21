#!/usr/bin/env python3
"""Run the Stage 1 grid across several GPUs with a simple worker pool.

Every job is an independent invocation of ``run_boundary_experiment.py`` — the
runs share no state, so the grid is embarrassingly parallel.  Two knobs matter:

``--gpus``              how many devices to spread across
``--workers-per-gpu``   how many concurrent runs per device

The second is worth more than it looks.  WideResNet-28-10 at 32x32 with batch
200 leaves a datacentre GPU badly under-occupied and the model is ~146 MB
against 48 GB of VRAM, so 2-3 workers per device typically buys most of another
factor of two.

Jobs whose ``<tag>.meta.json`` already exists are skipped, so the grid is
resumable and safe to re-invoke.  A failing job is recorded and the rest of the
grid continues; the exit status is non-zero if anything failed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optstate.adam_state import INTERVENTIONS   # noqa: E402

SEEDS = (0, 1, 2)
ORDERS = ("conventional", "perm1", "perm2", "perm3")
BETA1_VALUES = (0.0, 0.5, 0.99)          # 0.9 comes from the primary arm
LR_VALUES = (0.0003, 0.003)              # 1e-3 comes from the primary arm
SWEEP_DOMAINS = 8                        # 7 boundaries: the representative subset


@dataclass
class Job:
    tag: str
    stage: str
    argv: List[str]
    steps: int
    attempts: int = 0
    returncode: Optional[int] = None
    seconds: float = 0.0
    gpu: Optional[int] = None


def _primary_steps(domains: int, dbatch: int, branch: int) -> int:
    return domains * dbatch + (domains - 1) * (len(INTERVENTIONS) - 1) * branch


def _control_steps(domains: int, dbatch: int) -> int:
    half, tot = dbatch // 2, 0
    for i in range(domains):
        tot += half
        if i + 1 < domains:
            tot += len(INTERVENTIONS) * half
        tot += (len(INTERVENTIONS) - 1) * half + half
    return tot


def build_grid(args) -> List[Job]:
    """The preregistered grid, in the same decision-critical-first order the
    sequential driver uses (H1 -> H2 -> H4 -> everything else)."""
    jobs: List[Job] = []
    runner = str(ROOT / "scripts/run_boundary_experiment.py")
    common = ["--data-dir", args.data_dir, "--ckpt-dir", args.ckpt_dir,
              "--out", args.out, "--device", args.device]

    def add(stage, tag, extra, steps):
        jobs.append(Job(tag=tag, stage=stage, steps=steps,
                        argv=[runner, *extra, *common, "--tag", tag]))

    def boundary(stage, mode, order, seed, b1, lr, maxd):
        tag = f"{mode}_{order}_seed{seed}_b1{b1:g}_lr{lr:g}"
        steps = (_control_steps(maxd, args.domain_batches) if mode == "control"
                 else _primary_steps(maxd, args.domain_batches, args.branch_batches))
        add(stage, tag, ["--mode", mode, "--order", order, "--seed", str(seed),
                         "--beta1", str(b1), "--lr", str(lr),
                         "--max-domains", str(maxd)], steps)

    if "primary" in args.stages:
        for seed in SEEDS:                                   # 5a
            boundary("primary", "primary", "conventional", seed, 0.9, 1e-3, 15)
    if "control" in args.stages:
        for seed in SEEDS:                                   # 6
            boundary("control", "control", "conventional", seed, 0.9, 1e-3, 15)
    if "beta1" in args.stages:
        for b1 in BETA1_VALUES:                              # 7
            for seed in SEEDS:
                boundary("beta1", "primary", "conventional", seed, b1, 1e-3, SWEEP_DOMAINS)
    if "primary" in args.stages:
        for order in ORDERS:                                 # 5b
            if order == "conventional":
                continue
            for seed in SEEDS:
                boundary("primary", "primary", order, seed, 0.9, 1e-3, 15)
    if "lr" in args.stages:
        for lr in LR_VALUES:                                 # 8
            for seed in SEEDS:
                boundary("lr", "primary", "conventional", seed, 0.9, lr, SWEEP_DOMAINS)
    if "gradual" in args.stages:
        for seed in SEEDS:                                   # 9
            boundary("gradual", "gradual", "gradual", seed, 0.9, 1e-3, 5)
    if "sequence" in args.stages:
        for pol in INTERVENTIONS:                            # 9b, secondary
            for seed in SEEDS:
                tag = f"sequence_conventional_seed{seed}_{pol}"
                add("sequence", tag,
                    ["--mode", "sequence", "--sequence-policy", pol,
                     "--order", "conventional", "--seed", str(seed)],
                    15 * args.domain_batches)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--device", default="cuda",
                    help="device string handed to each run ('cuda', 'mps', 'cpu')")
    ap.add_argument("--stages", nargs="+",
                    default=["primary", "control", "beta1", "lr", "gradual", "sequence"])
    ap.add_argument("--domain-batches", type=int, default=50)
    ap.add_argument("--branch-batches", type=int, default=50)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "data/ckpt"))
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/raw/boundary"))
    ap.add_argument("--log-dir", default=str(ROOT / "logs/jobs"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_grid(args)
    todo = [j for j in jobs if not (out_dir / f"{j.tag}.meta.json").exists()]
    done_already = len(jobs) - len(todo)
    total_steps = sum(j.steps for j in todo)

    print(f"grid: {len(jobs)} runs, {done_already} already complete, "
          f"{len(todo)} to run, {total_steps:,} optimizer steps")
    by_stage: Dict[str, int] = {}
    for j in todo:
        by_stage[j.stage] = by_stage.get(j.stage, 0) + j.steps
    for k, v in by_stage.items():
        print(f"  {k:<10s} {sum(1 for j in todo if j.stage == k):>3d} runs  {v:>8,d} steps")
    slots = max(1, args.gpus) * max(1, args.workers_per_gpu)
    print(f"slots: {args.gpus} gpu(s) x {args.workers_per_gpu} worker(s) = {slots}")

    if args.dry_run:
        for j in todo:
            print(f"  [{j.stage}] {j.tag}  ({j.steps:,} steps)")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    running: List[tuple] = []          # (Job, Popen, fh, t0, slot)
    free_slots = list(range(slots))
    queue = list(todo)
    failed: List[Job] = []
    finished: List[Job] = []
    t_start = time.time()

    while queue or running:
        while queue and free_slots:
            slot = free_slots.pop(0)
            gpu = slot % max(1, args.gpus)
            job = queue.pop(0)
            job.gpu, job.attempts = gpu, job.attempts + 1
            env = dict(os.environ)
            if args.device.startswith("cuda"):
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            env["PYTHONWARNINGS"] = "ignore"
            fh = open(log_dir / f"{job.tag}.log", "a")
            fh.write(f"\n=== attempt {job.attempts} on gpu {gpu} ===\n"); fh.flush()
            proc = subprocess.Popen([args.python, "-W", "ignore", *job.argv],
                                    stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(ROOT.parent))
            running.append((job, proc, fh, time.time(), slot))
            print(f"[start] gpu{gpu} slot{slot} {job.tag}", flush=True)

        time.sleep(2.0)
        still = []
        for job, proc, fh, t0, slot in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc, fh, t0, slot))
                continue
            fh.close()
            job.returncode, job.seconds = rc, time.time() - t0
            free_slots.append(slot)
            if rc == 0:
                finished.append(job)
                print(f"[ok]    gpu{job.gpu} {job.tag}  {job.seconds/60:.1f} min "
                      f"({len(finished)}/{len(todo)} done, "
                      f"{(time.time()-t_start)/60:.0f} min elapsed)", flush=True)
            elif job.attempts <= args.retries:
                print(f"[retry] {job.tag} rc={rc}", flush=True)
                queue.append(job)
            else:
                failed.append(job)
                print(f"[FAIL]  {job.tag} rc={rc}  see {log_dir/(job.tag+'.log')}",
                      flush=True)
        running = still

    elapsed = time.time() - t_start
    summary = {
        "n_jobs": len(jobs), "already_complete": done_already,
        "ran": len(finished), "failed": [j.tag for j in failed],
        "elapsed_s": elapsed, "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "total_steps": total_steps,
        "seconds_per_step_aggregate": elapsed / total_steps if total_steps else None,
        "per_job_minutes": {j.tag: round(j.seconds / 60, 2) for j in finished},
    }
    (ROOT / "results/optimizer_state_stage1/raw/parallel_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\n{len(finished)} runs in {elapsed/3600:.2f} h "
          f"({elapsed/total_steps*1000:.0f} ms/step aggregate)")
    if failed:
        print("FAILED: " + ", ".join(j.tag for j in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
