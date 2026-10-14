"""Command-line interface: solve RCPSP instances with ABC / GS-ABC and follow the runs live.

Run ``python -m swarm --help`` for the available options.
"""

from __future__ import annotations

import argparse
import functools
import glob
import hashlib
import math
import multiprocessing as mp
import os
import queue
import random
import re
import signal
import statistics
import time
import traceback
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from swarm.abc import ABC, GSABC
from swarm.checkpoint import Checkpoint, CheckpointMismatch, load_snapshot, save_snapshot
from swarm.io import ResultInfo, SolutionInfo, save_results
from swarm.metrics import calculate_unique_individuals, jobs2times, positional_entropy
from swarm.problem import ProjectSchedulingModel
from swarm.rcpsp import RCPSPOperators

INSTANCE_NAME = re.compile(r"^j(30|60|90|120)(\d+)_(\d+)$")
REFERENCE_FILE = re.compile(r"^j\d+(opt|hrs|lb)$")
REPORT_INTERVAL = 0.25  # seconds between progress messages sent by a worker
REFRESH_INTERVAL = 0.125  # seconds between dashboard redraws
THROUGHPUT_WINDOW = 30.0  # seconds of history used to estimate evaluations per second
PAUSE_TIMEOUT = 60.0  # seconds to wait for running instances to save their progress after Ctrl+C
RESULT_SETTINGS = (  # options that determine the results, stored in checkpoints
    "algorithm",
    "init",
    "heuristics_rate",
    "sampling_rate",
    "population_size",
    "limit",
    "stagnation",
    "mutation_rate",
    "local_search_interval",
    "batch_onlookers",
    "max_evaluations",
    "stop_at_best_known",
    "seed",
)


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    init: str
    heuristics_rate: float
    sampling_rate: int
    population_size: int
    limit: int
    stagnation: int
    mutation_rate: float
    local_search_interval: int
    batch_onlookers: bool
    max_evaluations: int
    stop_at_best_known: bool
    seed: int
    log_every: int
    log_improvements: bool
    checkpoint_interval: float


@dataclass(frozen=True)
class RunTask:
    index: int
    path: str
    best_known: int
    seed: int
    snapshot: str | None = None  # where the run saves its progress, None without checkpoint

    @property
    def name(self) -> str:
        return Path(self.path).stem


# ----------------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm",
        description="Solve RCPSP instances (PSPLIB .sm files) with ABC or GS-ABC and follow the runs live.",
    )
    parser.add_argument(
        "paths", nargs="+", metavar="PATH", help="problem files, directories of .sm files or glob patterns"
    )

    algorithm = parser.add_argument_group("algorithm")
    algorithm.add_argument(
        "-a", "--algorithm", choices=("abc", "gs-abc"), default="abc", help="variant (default: %(default)s)"
    )
    algorithm.add_argument(
        "-p", "--population-size", type=int, default=10, metavar="N", help="food sources (default: %(default)s)"
    )
    algorithm.add_argument(
        "--limit", type=int, default=100, metavar="N", help="abandonment threshold for scout bees (default: %(default)s)"
    )
    algorithm.add_argument(
        "--stagnation", type=int, default=1, metavar="N", help="stagnation tracking window (default: %(default)s)"
    )
    algorithm.add_argument(
        "--mutation-rate", type=float, metavar="P", help="scout mutation probability, gs-abc only (default: 0.1)"
    )
    algorithm.add_argument(
        "--local-search-interval",
        type=int,
        default=-1,
        metavar="N",
        help="local search interval, -1 disables it (default: %(default)s)",
    )
    algorithm.add_argument(
        "--batch-onlookers", action="store_true", help="evaluate all the onlookers of a cycle in one batch"
    )

    init = parser.add_argument_group("initialization")
    init.add_argument(
        "-i",
        "--init",
        choices=("random", "heuristic", "mcmc"),
        default="random",
        help="random orders, priority-rule orders (LDF, SDF, MRF, LRF) or MCMC samples around them "
        "(default: %(default)s)",
    )
    init.add_argument(
        "--heuristics-rate", type=float, metavar="R", help="share of the colony seeded from priority rules (default: 1.0)"
    )
    init.add_argument("--sampling-rate", type=int, metavar="N", help="MCMC spacing between samples (default: 25)")

    run = parser.add_argument_group("run")
    run.add_argument(
        "-e", "--max-evaluations", type=int, default=50000, metavar="N", help="evaluation budget per run (default: %(default)s)"
    )
    run.add_argument(
        "--stop-at-best-known",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop a run as soon as it reaches the best-known makespan",
    )
    run.add_argument(
        "--best-known-dir",
        type=Path,
        metavar="DIR",
        help="directory with the j*opt.sm / j*hrs.sm files (default: next to the problems)",
    )
    run.add_argument(
        "-s",
        "--seed",
        type=int,
        help="base seed; each instance runs with its own seed derived from it and the instance name "
        "(default: random, shown in the header)",
    )
    run.add_argument(
        "-w", "--workers", type=int, default=1, metavar="N", help="runs in parallel, 0 = one per CPU core (default: %(default)s)"
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "-o", "--output", type=Path, metavar="FILE", help="save the results as JSON (readable with swarm.load_results)"
    )
    output.add_argument(
        "--log-every", type=int, default=0, metavar="N", help="also log every N-th generation (default: only improvements)"
    )
    output.add_argument("-q", "--quiet", action="store_true", help="only log finished runs")

    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        metavar="DIR",
        help="save the progress in DIR; running the same command again resumes from it",
    )
    checkpoint.add_argument(
        "--checkpoint-interval",
        type=float,
        metavar="SECONDS",
        help="seconds between snapshots of the running instances (default: 30)",
    )
    return parser


def parse_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[RunConfig, int]:
    errors = [
        (args.workers < 0, "--workers must be >= 0"),
        (args.max_evaluations < 1, "--max-evaluations must be >= 1"),
        (args.population_size < 1, "--population-size must be >= 1"),
        (args.limit < 0, "--limit must be >= 0"),
        (args.stagnation < 1, "--stagnation must be >= 1"),
        (args.local_search_interval == 0 or args.local_search_interval < -1, "--local-search-interval must be -1 or >= 1"),
        (args.mutation_rate is not None and args.algorithm != "gs-abc", "--mutation-rate only applies to --algorithm gs-abc"),
        (args.mutation_rate is not None and not 0 <= args.mutation_rate <= 1, "--mutation-rate must be between 0 and 1"),
        (args.heuristics_rate is not None and args.init == "random", "--heuristics-rate requires --init heuristic or mcmc"),
        (args.heuristics_rate is not None and not 0 <= args.heuristics_rate <= 1, "--heuristics-rate must be between 0 and 1"),
        (args.sampling_rate is not None and args.init != "mcmc", "--sampling-rate requires --init mcmc"),
        (args.sampling_rate is not None and args.sampling_rate < 1, "--sampling-rate must be >= 1"),
        (args.seed is not None and not 0 <= args.seed < 2**32, "--seed must be between 0 and 2**32 - 1"),
        (args.log_every < 0, "--log-every must be >= 0"),
        (args.quiet and args.log_every > 0, "--quiet and --log-every cannot be combined"),
        (args.checkpoint_interval is not None and args.checkpoint is None, "--checkpoint-interval requires --checkpoint"),
        (args.checkpoint_interval is not None and args.checkpoint_interval <= 0, "--checkpoint-interval must be > 0"),
    ]
    for failed, message in errors:
        if failed:
            parser.error(message)

    if args.init == "random":
        heuristics_rate, sampling_rate = 0.0, 0
    else:
        heuristics_rate = 1.0 if args.heuristics_rate is None else args.heuristics_rate
        sampling_rate = 0 if args.init == "heuristic" else 25 if args.sampling_rate is None else args.sampling_rate

    config = RunConfig(
        algorithm=args.algorithm,
        init=args.init,
        heuristics_rate=heuristics_rate,
        sampling_rate=sampling_rate,
        population_size=args.population_size,
        limit=args.limit,
        stagnation=args.stagnation,
        mutation_rate=0.1 if args.mutation_rate is None else args.mutation_rate,
        local_search_interval=args.local_search_interval,
        batch_onlookers=args.batch_onlookers,
        max_evaluations=args.max_evaluations,
        stop_at_best_known=args.stop_at_best_known,
        seed=random.SystemRandom().randrange(2**32) if args.seed is None else args.seed,
        log_every=args.log_every,
        log_improvements=not args.quiet,
        checkpoint_interval=30.0 if args.checkpoint_interval is None else args.checkpoint_interval,
    )
    return config, args.workers or os.cpu_count() or 1


# ------------------------------------------------------------------------------ problems
def find_problem_files(patterns: list[str]) -> list[Path]:
    found: dict[Path, None] = {}
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            candidates = [path]
        else:
            matches = path.glob("*.sm") if path.is_dir() else map(Path, glob.glob(pattern))
            candidates = [p for p in matches if p.is_file() and not REFERENCE_FILE.match(p.stem)]
        if not candidates:
            raise FileNotFoundError(f"no problem files found for '{pattern}'")
        found.update(dict.fromkeys(p.resolve() for p in candidates))
    return sorted(found, key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])


def instance_seed(seed: int, name: str) -> int:
    """Seed for the run on instance ``name``: an independent stream per instance, reproducible from ``seed``."""
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return int(np.random.SeedSequence([seed, key]).generate_state(1)[0])


def best_known_makespan(problem: Path, search_dirs: list[Path]) -> int:
    """Best-known makespan from the PSPLIB j*opt.sm / j*hrs.sm files, or 0 when unknown."""
    match = INSTANCE_NAME.match(problem.stem)
    if not match:
        return 0
    size, parameter, instance = (int(group) for group in match.groups())
    for directory in [*search_dirs, problem.parent, problem.parent.parent]:
        for kind in ("opt", "hrs"):
            value = _reference_table(directory / f"j{size}{kind}.sm").get((parameter, instance))
            if value is not None:
                return value
    return 0


@functools.cache
def _reference_table(path: Path) -> dict[tuple[int, int], int]:
    table: dict[tuple[int, int], int] = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 3 and all(f.isdigit() for f in fields[:3]):
                table[int(fields[0]), int(fields[1])] = int(fields[2])
    return table


# ------------------------------------------------------------------------------- workers
_updates: Any = None
_stop: Any = None


def _init_worker(updates: Any, stop: Any) -> None:
    global _updates, _stop
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    _updates, _stop = updates, stop


def _solve(task: RunTask, config: RunConfig) -> None:
    if _stop.is_set():
        return  # the batch is pausing: queued instances are left for the next session
    try:
        kind, payload = _run(task, config)
        _updates.put((kind, task.index, payload))
    except Exception:
        _updates.put(("error", task.index, traceback.format_exc()))


@dataclass
class RunProgress:
    """Everything an unfinished run needs to continue; pickled into checkpoint snapshots."""

    problem_id: str
    optimizer: ABC
    best_history: list[int] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    unique: list[int] = field(default_factory=list)
    scouts: list[int] = field(default_factory=list)
    last_improvement: int = 0
    elapsed: float = 0.0


def _new_run(task: RunTask, config: RunConfig) -> RunProgress:
    model = ProjectSchedulingModel.from_file(task.path)
    model.best_known = task.best_known
    operators = RCPSPOperators(model, heuristics_rate=config.heuristics_rate, sampling_rate=config.sampling_rate)
    options = dict(
        population_size=config.population_size,
        limit=config.limit,
        stagnation=config.stagnation,
        local_search_interval=config.local_search_interval,
        batch_onlookers=config.batch_onlookers,
        seed=task.seed,
    )
    if config.algorithm == "gs-abc":
        return RunProgress(model.name, GSABC(operators, mutation_rate=config.mutation_rate, **options))
    return RunProgress(model.name, ABC(operators, **options))


def _resumed_run(task: RunTask) -> RunProgress | None:
    snapshot = load_snapshot(task.snapshot) if task.snapshot else None
    if snapshot is None or (snapshot["seed"], snapshot["best_known"]) != (task.seed, task.best_known):
        return None
    random.setstate(snapshot["random"])
    np.random.set_state(snapshot["numpy"])
    return snapshot["progress"]


def _run(task: RunTask, config: RunConfig) -> tuple[str, dict]:
    started = time.perf_counter()
    progress = _resumed_run(task)
    resumed = progress is not None
    if progress is None:
        progress = _new_run(task, config)
    optimizer = progress.optimizer
    resumed_at = {"generation": optimizer.cycle, "evaluations": optimizer.n_evaluations} if resumed else None
    _updates.put(("start", task.index, resumed_at))
    target = task.best_known if config.stop_at_best_known and task.best_known > 0 else -math.inf

    def elapsed() -> float:
        return progress.elapsed + time.perf_counter() - started

    def step() -> None:
        schedules = optimizer.ask()
        optimizer.tell([s.get_makespan() for s in schedules])

    def record(released: int) -> None:
        progress.best_history.append(int(optimizer.best_fitness))
        progress.diversity.append(
            float(positional_entropy([[job.id for job in s.jobs] for s in optimizer.population])[1])
        )
        progress.unique.append(calculate_unique_individuals(optimizer.population))
        progress.scouts.append(released)

    def send(log: bool, improved: bool) -> None:
        stats = dict(
            generation=optimizer.cycle,
            evaluations=optimizer.n_evaluations,
            best=int(optimizer.best_fitness),
            mean=float(np.mean(optimizer.fitness)),
            diversity=progress.diversity[-1],
            scouts=sum(progress.scouts),
            last_improvement=progress.last_improvement,
            elapsed=elapsed(),
            log=log,
            improved=improved,
        )
        _updates.put(("progress", task.index, stats))

    def save() -> None:
        nonlocal started
        now = time.perf_counter()
        progress.elapsed += now - started
        started = now
        state = dict(
            seed=task.seed,
            best_known=task.best_known,
            progress=progress,
            random=random.getstate(),
            numpy=np.random.get_state(),
        )
        save_snapshot(task.snapshot, state)

    if not resumed:
        step()
        record(0)
    send(log=config.log_improvements and not resumed, improved=False)
    generation, rounds = optimizer.cycle, 0
    last_report = last_save = time.perf_counter()
    while optimizer.n_evaluations < config.max_evaluations and optimizer.best_fitness > target:
        if task.snapshot and rounds == 0:  # snapshots are taken between generations
            pausing = _stop.is_set()
            if pausing or time.perf_counter() - last_save >= config.checkpoint_interval:
                save()
                last_save = time.perf_counter()
                if pausing:
                    return "paused", {"generation": optimizer.cycle, "evaluations": optimizer.n_evaluations}
        step()
        rounds += 1
        if optimizer.cycle == generation:
            continue
        generation, rounds = optimizer.cycle, 0
        improved = optimizer.best_fitness < progress.best_history[-1]
        if improved:
            progress.last_improvement = generation
        record(optimizer.n_scouts)
        log = (improved and config.log_improvements) or (config.log_every > 0 and generation % config.log_every == 0)
        now = time.perf_counter()
        if log or now - last_report >= REPORT_INTERVAL:
            send(log, improved)
            last_report = now
    if rounds:
        record(0)  # the run stopped in the middle of a generation

    best = optimizer.best
    return "done", dict(
        best=int(optimizer.best_fitness),
        generations=optimizer.cycle,
        evaluations=optimizer.n_evaluations,
        elapsed=elapsed(),
        reason="best known reached" if optimizer.best_fitness <= target else "budget exhausted",
        result=ResultInfo(
            problem_id=progress.problem_id,
            best=SolutionInfo(makespan=int(optimizer.best_fitness), start_times=jobs2times(best.jobs)),
            best_history=progress.best_history,
            population_diversity=progress.diversity,
            unique_solutions=progress.unique,
            scout_bees=progress.scouts,
        ),
    )


# ----------------------------------------------------------------------------- dashboard
@dataclass
class RunState:
    name: str
    best_known: int
    generation: int = 0
    evaluations: int = 0
    best: int | None = None
    mean: float = math.nan
    diversity: float = math.nan
    scouts: int = 0
    last_improvement: int = 0
    elapsed: float = 0.0
    received: float = field(default_factory=time.monotonic)


class Dashboard:
    """Main-process view of all runs, rendered live by ``rich``."""

    def __init__(
        self,
        config: RunConfig,
        tasks: list[RunTask],
        workers: int,
        restored: dict[int, dict] | None = None,
        checkpoint: Path | None = None,
        resumable: int = 0,
    ):
        self.config = config
        self.tasks = tasks
        self.workers = workers
        self.checkpoint = checkpoint
        self.resumable = resumable
        self.active: dict[int, RunState] = {}
        self.finished: dict[int, dict] = {}
        self.failed: dict[int, str] = {}
        self.paused: dict[int, dict] = {}
        self.closed_evaluations = 0
        self.unused_budget = 0
        self.restored_evaluations = 0  # done in earlier sessions, excluded from the throughput
        self.restored_runs = len(restored or {})
        for index, outcome in (restored or {}).items():
            self.finished[index] = outcome
            self.closed_evaluations += outcome["evaluations"]
            self.unused_budget += config.max_evaluations - outcome["evaluations"]
            self.restored_evaluations += outcome["evaluations"]
        self.started = time.monotonic()
        self._samples: deque[tuple[float, int]] = deque()
        self._spinner = Spinner("dots", style="cyan")

    @property
    def done(self) -> bool:
        return len(self.finished) + len(self.failed) == len(self.tasks)

    def evaluations(self) -> int:
        return self.closed_evaluations + sum(run.evaluations for run in self.active.values())

    def handle(self, kind: str, index: int, payload: Any) -> list[Text]:
        """Apply a worker message and return the lines to log above the live view."""
        task = self.tasks[index]
        if kind == "start":
            run = self.active[index] = RunState(task.name, task.best_known)
            if payload is None:
                return []
            run.generation, run.evaluations = payload["generation"], payload["evaluations"]
            self.restored_evaluations += payload["evaluations"]
            return [_resumed_line(task, payload)]
        if kind == "progress":
            run = self.active[index]
            for key in ("generation", "evaluations", "best", "mean", "diversity", "scouts", "last_improvement", "elapsed"):
                setattr(run, key, payload[key])
            run.received = time.monotonic()
            return [_generation_line(run, payload["improved"])] if payload["log"] else []

        run = self.active.pop(index, None)
        if kind == "paused":
            self.paused[index] = payload
            self.closed_evaluations += payload["evaluations"]
            return [_paused_line(task, payload)]
        used = payload["evaluations"] if kind == "done" else run.evaluations if run else 0
        self.closed_evaluations += used
        self.unused_budget += self.config.max_evaluations - used
        if kind == "done":
            self.finished[index] = payload
            return [_finished_line(task, payload)]
        self.failed[index] = payload
        return [_failed_line(task, payload)]

    def header(self) -> Panel:
        c = self.config
        initialization = {
            "random": "random activity orders",
            "heuristic": f"priority rules (heuristics rate {c.heuristics_rate})",
            "mcmc": f"MCMC around priority rules (heuristics rate {c.heuristics_rate}, spacing {c.sampling_rate})",
        }[c.init]
        local_search = "off" if c.local_search_interval == -1 else f"interval {c.local_search_interval}"
        lines = [
            f"{len(self.tasks)} problem(s) · {self.workers} worker(s) · seed {c.seed} (derived per instance) · "
            f"budget {c.max_evaluations:,} evaluations" + (" · stop at best known" if c.stop_at_best_known else ""),
            f"population {c.population_size} · limit {c.limit} · stagnation {c.stagnation} · local search {local_search}"
            + (f" · mutation rate {c.mutation_rate:g}" if c.algorithm == "gs-abc" else "")
            + (" · batch onlookers" if c.batch_onlookers else ""),
            f"initialization: {initialization}",
        ]
        if self.checkpoint:
            earlier = f"{self.restored_runs} finished earlier · {self.resumable} to resume"
            lines.append(f"checkpoint {self.checkpoint} ({earlier if self.restored_runs or self.resumable else 'new'})")
        title = f"RCPSP · {'GS-ABC' if c.algorithm == 'gs-abc' else 'ABC'}"
        return Panel("\n".join(lines), title=title, title_align="left", border_style="cyan", expand=False)

    def summary(self, output: Path | None, interrupted: bool) -> Panel:
        outcomes = {i: self.finished[i] for i in sorted(self.finished)}
        gaps = [
            gap for i, outcome in outcomes.items() if (gap := _gap(outcome["best"], self.tasks[i].best_known)) is not None
        ]
        runs = f"{len(outcomes)}/{len(self.tasks)} finished"
        if self.restored_runs:
            runs += f" ({self.restored_runs} in earlier sessions)"
        if self.failed:
            runs += f" · {len(self.failed)} failed"
        if self.paused:
            runs += f" · {len(self.paused)} saved to resume"
        if interrupted and self.active:
            runs += f" · {len(self.active)} interrupted"
        lines = [f"runs               {runs}"]
        if gaps:
            reached = sum(gap <= 0 for gap in gaps)
            lines.append(f"best known reached {reached}/{len(gaps)} ({reached / len(gaps):.1%})")
            lines.append(f"gap to best known  mean {statistics.fmean(gaps):+.2f}% · max {max(gaps):+.2f}%")
        if outcomes:
            evaluations = sum(outcome["evaluations"] for outcome in outcomes.values())
            lines.append(f"evaluations        {evaluations:,} (mean {evaluations / len(outcomes):,.0f} per run)")
        lines.append(f"wall time          {_duration(time.monotonic() - self.started)}")
        if output:
            lines.append(f"results            {output}")
        if self.checkpoint and not self.done:
            lines.append(f"checkpoint         {self.checkpoint} (run the same command to resume)")
        failed = interrupted or bool(self.failed)
        return Panel(
            "\n".join(lines),
            title="Interrupted" if interrupted else "Summary",
            title_align="left",
            border_style="yellow" if failed else "green",
            expand=False,
        )

    def __rich__(self) -> RenderableType:
        return Group(self._runs_table(), self._overall())

    def _runs_table(self) -> RenderableType:
        if not self.active:
            return Text("Waiting for workers…", style="dim")
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False)
        table.add_column("Problem", style="bold", no_wrap=True)
        for name in ("Gen", "Evaluations", "Best", "Target", "Gap", "Mean", "Div", "Scouts", "Stall", "Elapsed", "ETA"):
            table.add_column(name, justify="left" if name == "Evaluations" else "right", no_wrap=True)
        now = time.monotonic()
        budget = self.config.max_evaluations
        for run in self.active.values():
            elapsed = run.elapsed + now - run.received
            eta = max(0.0, (budget - run.evaluations) * run.elapsed / run.evaluations - (now - run.received)) if run.evaluations else None
            evaluations = Table.grid(padding=(0, 1))
            evaluations.add_row(ProgressBar(total=budget, completed=min(run.evaluations, budget), width=12), f"{run.evaluations:>7,}")
            table.add_row(
                run.name,
                f"{run.generation:,}",
                evaluations,
                "—" if run.best is None else str(run.best),
                str(run.best_known) if run.best_known else "—",
                _gap_text(_gap(run.best, run.best_known)),
                f"{run.mean:.1f}",
                f"{run.diversity:.2f}",
                f"{run.scouts:,}",
                f"{run.generation - run.last_improvement:,}",
                _duration(elapsed),
                _duration(eta),
            )
        return table

    def _overall(self) -> RenderableType:
        now, done = time.monotonic(), self.evaluations()
        self._samples.append((now, done - self.restored_evaluations))
        while now - self._samples[0][0] > THROUGHPUT_WINDOW:
            self._samples.popleft()
        first_time, first_done = self._samples[0]
        rate = (self._samples[-1][1] - first_done) / (now - first_time) if now - first_time >= 1 else None
        total = max(len(self.tasks) * self.config.max_evaluations - self.unused_budget, 1)
        eta = (total - done) / rate if rate else None
        closed = len(self.finished) + len(self.failed)
        queued = len(self.tasks) - closed - len(self.active)

        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_column(no_wrap=True)
        grid.add_row(
            self._spinner,
            Text(f"{closed}/{len(self.tasks)} runs", style="bold"),
            ProgressBar(total=total, completed=min(done, total)),
            f"{done / total:6.1%} · {len(self.active)} running · {queued} queued · {rate or 0:,.0f} evals/s · "
            f"elapsed {_duration(now - self.started)} · ETA {_duration(eta)}",
        )
        return grid


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "-:--:--"
    return str(timedelta(seconds=round(seconds)))


def _gap(best: int | None, best_known: int) -> float | None:
    if best is None or best_known <= 0:
        return None
    return (best - best_known) / best_known * 100


def _gap_text(gap: float | None) -> Text:
    if gap is None:
        return Text("—", style="dim")
    style = "bold green" if gap <= 0 else "yellow" if gap < 5 else "red"
    return Text(f"{gap:+.2f}%", style=style)


def _line_start(name: str) -> Text:
    line = Text(datetime.now().strftime("%H:%M:%S "), style="dim")
    line.append(f"{name:<10}", style="bold")
    return line


def _generation_line(run: RunState, improved: bool) -> Text:
    line = _line_start(run.name)
    line.append(f" gen {run.generation:>6,}  evals {run.evaluations:>7,}  best {run.best:>4} ")
    line.append_text(_gap_text(_gap(run.best, run.best_known)))
    line.append(f"  mean {run.mean:7.1f}  div {run.diversity:.2f}  scouts {run.scouts:,}")
    if run.generation == 0:
        line.append("  initial population", style="cyan")
    elif improved:
        line.append("  ▼ new best", style="green")
    return line


def _finished_line(task: RunTask, outcome: dict) -> Text:
    gap = _gap(outcome["best"], task.best_known)
    line = _line_start(task.name)
    line.append(" ✓ finished" if gap is not None and gap <= 0 else " • finished", style="bold green")
    line.append(f"  best {outcome['best']} ")
    line.append_text(_gap_text(gap))
    line.append(
        f"  {outcome['generations']:,} generations · {outcome['evaluations']:,} evals · "
        f"{_duration(outcome['elapsed'])} · {outcome['reason']}"
    )
    return line


def _failed_line(task: RunTask, error: str) -> Text:
    line = _line_start(task.name)
    line.append(" ✗ failed\n", style="bold red")
    line.append(error.rstrip(), style="red")
    return line


def _resumed_line(task: RunTask, resumed_at: dict) -> Text:
    line = _line_start(task.name)
    line.append(" ↻ resumed", style="bold cyan")
    line.append(f"  from the checkpoint at generation {resumed_at['generation']:,} · {resumed_at['evaluations']:,} evals")
    return line


def _paused_line(task: RunTask, paused_at: dict) -> Text:
    line = _line_start(task.name)
    line.append(" ⏸ saved", style="bold yellow")
    line.append(f"  to the checkpoint at generation {paused_at['generation']:,} · {paused_at['evaluations']:,} evals")
    return line


# ---------------------------------------------------------------------------------- main
def _monitor(
    dashboard: Dashboard,
    updates: Any,
    console: Console,
    until: Callable[[], bool],
    on_message: Callable[[str, int, Any], None],
    timeout: float = math.inf,
) -> None:
    deadline = time.monotonic() + timeout
    with Live(dashboard, console=console, auto_refresh=False, transient=True) as live:
        next_refresh = 0.0
        while not until() and time.monotonic() < deadline:
            messages = []
            try:
                messages.append(updates.get(timeout=REFRESH_INTERVAL))
                while len(messages) < 200:
                    messages.append(updates.get_nowait())
            except queue.Empty:
                pass
            lines: list[Text] = []
            for message in messages:
                on_message(*message)
                lines += dashboard.handle(*message)
            if lines:
                live.console.print(Group(*lines))
            if time.monotonic() >= next_refresh:
                live.refresh()
                next_refresh = time.monotonic() + REFRESH_INTERVAL


def _run_batch(
    tasks: list[RunTask], config: RunConfig, workers: int, dashboard: Dashboard, console: Console, checkpoint: Checkpoint | None
) -> bool:
    """Run ``tasks`` in a process pool; returns True when interrupted with Ctrl+C."""

    def on_message(kind: str, index: int, payload: Any) -> None:
        if kind == "done" and checkpoint is not None:
            checkpoint.record(dashboard.tasks[index].name, payload)

    context = mp.get_context("spawn")
    updates, stop = context.Queue(), context.Event()
    pool = context.Pool(workers, initializer=_init_worker, initargs=(updates, stop))
    try:
        for task in tasks:
            pool.apply_async(_solve, (task, config))
        pool.close()
        _monitor(dashboard, updates, console, lambda: dashboard.done, on_message)
        pool.join()
        return False
    except KeyboardInterrupt:
        if checkpoint is not None and dashboard.active:
            stop.set()
            console.print("Saving the running instances to the checkpoint… (Ctrl+C again to quit now)", style="yellow")
            try:
                _monitor(dashboard, updates, console, lambda: not dashboard.active, on_message, PAUSE_TIMEOUT)
            except KeyboardInterrupt:
                pass
        pool.terminate()
        pool.join()
        return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config, workers = parse_config(parser, args)
    try:
        files = find_problem_files(args.paths)
    except FileNotFoundError as error:
        parser.error(str(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    finished_before: dict[str, dict] = {}
    if args.checkpoint:
        try:
            stored = Checkpoint.stored_settings(args.checkpoint)
            if stored is not None and args.seed is None:
                config = replace(config, seed=stored["seed"])  # resuming without --seed keeps the original seed
            checkpoint = Checkpoint.open(args.checkpoint, {name: getattr(config, name) for name in RESULT_SETTINGS})
        except CheckpointMismatch as error:
            parser.error(str(error))
        finished_before = checkpoint.finished_runs()

    search_dirs = [args.best_known_dir] if args.best_known_dir else []
    tasks = [
        RunTask(
            i,
            str(path),
            best_known_makespan(path, search_dirs),
            instance_seed(config.seed, path.stem),
            str(checkpoint.snapshot_path(path.stem)) if checkpoint else None,
        )
        for i, path in enumerate(files)
    ]
    restored = {task.index: finished_before[task.name] for task in tasks if task.name in finished_before}
    pending = [task for task in tasks if task.index not in restored]
    resumable = sum(1 for task in pending if task.snapshot and Path(task.snapshot).exists())
    workers = max(1, min(workers, len(pending)))

    console = Console()
    dashboard = Dashboard(config, tasks, workers, restored, args.checkpoint, resumable)
    console.print(dashboard.header())
    interrupted = _run_batch(pending, config, workers, dashboard, console, checkpoint) if pending else False

    results = [dashboard.finished[i]["result"] for i in sorted(dashboard.finished)]
    output = args.output if args.output and results else None
    if output:
        save_results(results, str(output))
    console.print(dashboard.summary(output, interrupted))
    return 130 if interrupted else 1 if dashboard.failed else 0
