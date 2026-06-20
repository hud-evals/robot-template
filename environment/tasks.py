"""LIBERO taskset: one episode per row, graded binary; success_rate = mean reward.

`make_tasks` is the single source of task rows — `run.py` and every `examples/`
script build their rollouts from it, so they all evaluate the same thing. Each row
is `Task(env="libero", id=<suite>, args={libero_task_id, init_state_id})`; the env
(`env.py`) exposes one `@env.template` per suite.

Run this file to (re)write `full_bench_tasks.json`, the full sweep grouped by suite::

    python environment/tasks.py

Grid (validated against the installed LIBERO suites): every task has exactly 50
saved init states; spatial/object/goal/10 hold 10 tasks each, libero_90 holds 90 —
130 tasks x 50 = 6500 episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

from hud.eval import Task

ENV = "libero"

# suite -> number of tasks (validated). Each task has exactly 50 saved init states.
SUITES: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
INIT_STATES_PER_TASK = 50

OUTPUT = Path(__file__).resolve().parent / "full_bench_tasks.json"


def make_tasks(suite: str = "libero_spatial", n: int = 3, init_state_id: int = 0) -> list[Task]:
    """`n` episodes from one suite (task ids 0..n-1) at a fixed init state."""
    return [
        Task(env=ENV, id=suite, args={"libero_task_id": t, "init_state_id": init_state_id})
        for t in range(n)
    ]


def load_bench(suite: str | None = None) -> list[Task]:
    """Every episode of the full benchmark from ``full_bench_tasks.json``.

    ``suite=None`` returns all 6500 rows; pass a suite name for just its 500 (4500
    for libero_90). Run ``python environment/tasks.py`` to (re)generate the file.
    """
    if not OUTPUT.exists():
        raise FileNotFoundError(f"{OUTPUT} missing — run `python environment/tasks.py` to generate it.")
    grouped = json.loads(OUTPUT.read_text())
    rows = grouped[suite] if suite is not None else [r for suite_rows in grouped.values() for r in suite_rows]
    return [Task(env=r["env"], id=r["id"], args=r["args"]) for r in rows]


# The default rollouts the runners use out of the box (small, fast to smoke-test).
tasks = make_tasks()


def export(path: Path = OUTPUT) -> Path:
    """Write the full sweep as ``{suite: [row, ...]}`` (rows are plain Task dicts)."""
    grouped = {
        suite: [
            {"env": ENV, "id": suite, "args": {"libero_task_id": t, "init_state_id": i}}
            for t in range(n_tasks)
            for i in range(INIT_STATES_PER_TASK)
        ]
        for suite, n_tasks in SUITES.items()
    }
    Path(path).write_text(json.dumps(grouped, indent=2) + "\n", encoding="utf-8")
    total = sum(len(rows) for rows in grouped.values())
    print(f"wrote {total} tasks across {len(grouped)} suites -> {path}")
    return path


if __name__ == "__main__":
    export()
