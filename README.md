# HUD Robot Template (v6)

A HUD v6 robot environment built like every other [hud-evals template](https://github.com/hud-evals):
a `Dockerfile.hud` + `pyproject.toml` make `environment/` a deployable env, and one
central run drives it. The worked example is **SmolVLA × LIBERO** — a VLA policy
picking and placing in simulation, scored as a success rate.

For the framework concepts behind it, read the docs:
**[https://docs.hud.ai/v6/core/robots](https://docs.hud.ai/v6/core/robots)**.

> The robot framework is in `beta` — we invite you to build on it, test it, and break it.

## What you get

- **One central run** — `run.py`: SmolVLA × LIBERO, end to end. Flags switch where the
  policy and env run (local / batched / remote); the thing evaluated never changes.
- **A deployable env** — `environment/` (the LIBERO sim), packaged by `Dockerfile.hud`
  and registered by name with `hud deploy`. The same image runs under Docker, Modal,
  Daytona, or the HUD platform.
- **An inventory** of reusable agents and envs to copy: more VLA policies, more
  simulators, remote policy servers, and fill-in-the-blank scaffolds.
- **A skill** (`skills/robot-integration/`) that teaches an agent to integrate a new
  model or env against this template.

## Layout

```
robot-template/
├── Dockerfile.hud            ← builds the LIBERO env image (hud build / hud deploy find it by name)
├── pyproject.toml            ← harness deps; with Dockerfile.hud, marks this a deployable HUD env
├── run.py                    ← THE central run: SmolVLA × LIBERO (--batched / --remote / --modal)
├── test_install.py           ← verify your environment is ready
├── QUICKSTART.md             ← install + run in ~5 min
├── environment/              ← the deployed env (the only code dir `hud deploy` ships)
│   ├── env.py                   Environment(name="libero"); served as `environment/env:env`
│   ├── tasks.py                 the taskset — make_tasks() / load_bench(); run.py builds from it
│   ├── full_bench_tasks.json    the full 6500-episode sweep, grouped by suite (`run.py --full`)
│   ├── config.py                contract selection + scenarios
│   ├── libero_sim_bridge.py     reset / step / get_observation against the LIBERO sim
│   ├── env_streamer.py          MJPEG live-view sink
│   ├── utilities.py             small math helpers
│   └── libero_ee_{del,abs}.json the wire contracts (delta / absolute eef)
├── examples/                 ← the three run modes, one file each (all import the same agent)
│   ├── local.py                 policy here, env in Docker
│   ├── batched.py               many episodes off one batched GPU forward
│   └── remote.py                policy served on a GPU + env on Modal; laptop = harness
├── inventory/                ← the reusable library (copy from here)
│   ├── agents/                  policies (the agent side)
│   │   ├── smolvla_libero.py       the one shared agent run.py + examples use
│   │   ├── pi05_libero.py          a bigger VLA, same surface
│   │   ├── vla_jepa_libero.py      a chunked policy
│   │   ├── scaffold/               ← fill-in template for your own agent
│   │   └── remote/                 ← serve a model on a GPU, drive it weightlessly
│   └── envs/                   more simulators to copy (the env side)
│       ├── simpler/                another sync sim (WidowX / ManiSkill3)
│       ├── robolab/                main-thread sim (Isaac): sim_process.py + RobotEndpoint.remote
│       ├── scaffold/               ← fill-in env: env.py + bridge + minimal contract
│       └── remote/                 ← build + serve an env on Modal or Daytona
└── skills/robot-integration/ ← agent skill for adding a new model or env
```

The library envs under `inventory/envs/` ship a `docker.md` recipe instead of a
`Dockerfile.hud`, because only the **one** env at `environment/` is the deployable
target. Promote any of them the same way LIBERO is set up here: move it to
`environment/` and point `Dockerfile.hud`'s `hud serve` at it.

## Run it

```bash
python run.py                       # local: policy here, LIBERO in Docker
python run.py --batched 8           # 8 episodes off one batched GPU forward
python run.py --full                # the whole libero_spatial suite (500 episodes)
python run.py --full --suite all    # the entire 6500-episode benchmark
python run.py --remote HOST:PORT    # policy served on a GPU box; env in Docker
python run.py --remote HOST:PORT --modal   # ...and the env in a Modal sandbox
```

`--full` runs every episode from `environment/full_bench_tasks.json` — the full LIBERO
sweep (130 tasks × 50 init states = 6500 episodes), grouped by suite. Pair it with
`--batched` for throughput. Regenerate the file with `python environment/tasks.py`.

See **[QUICKSTART.md](QUICKSTART.md)** for install + building the env image. To host the
env on the HUD platform instead of running it yourself:

```bash
hud deploy                          # builds Dockerfile.hud, registers the env as "libero"
```

## What the abstractions give you (for free)

You implement small named seams; the framework owns everything between them.

**Agent side** — subclass `RobotAgent`, set a `Model` and an `Adapter`:

- **`RobotAgent`** — the episode loop, the openpi/0 wire protocol, and **automatic
  per-step telemetry** streamed to the HUD platform (every camera frame + executed
  action, replayable in the trace viewer). Zero config.
- **`LeRobotModel`** — runs a stock LeRobot checkpoint (preprocess → forward →
  postprocess); a complete agent is ~15 lines (`inventory/agents/smolvla_libero.py`).
- **`RemoteModel(host, port)`** — the same agent, but the weights live on a GPU box and
  only a stateless chunk forward crosses the network. Point it at any openpi server
  (your own, NVIDIA Cosmos, a hosted pi0.5).
- **`BatchedAgent`** — wrap any agent to run N rollouts concurrently off **one batched
  GPU forward**. You change nothing else.
- **`LeRobotAdapter` / `OpenPIAdapter`** — env↔policy space translation, learned from the
  env's contract (no shared config, no env-specific keys in your agent).

**Env side** — subclass `RobotBridge` (`reset` / `step` / `get_observation`) and write a
declarative `env.py`:

- **`RobotBridge`** — owns the WebSocket serve loop + single-agent connection; you just
  step your sim.
- **`RobotEndpoint`** — one control handle (`start`/`reset`/`result`/`stop`), identical
  whether the sim is in-process or in **another process** (`.remote(...)`, for
  Isaac/Omniverse — see `inventory/envs/robolab/`).
- **`SimRunner`** — one-line choice of which thread runs a thread-affine sim
  (`Inline` / `Thread` / `MainThread`).

**Record datasets, for free:** set `HUD_RECORD_DIR` to also record every
`(observation, action)` tick into a **LeRobot v3 dataset** — the rollouts you just ran,
ready to finetune on. Push to the Hub with `HUD_HF_REPO` + `HF_TOKEN`. Details:
[docs](https://docs.hud.ai/v6/core/robots#recording-datasets).

## Local vs. remote

`run.py` (and `examples/local.py`) runs the policy **in your process** — simplest, but it
wants a local GPU. `--remote` (and `examples/remote.py`) keeps the policy's weights on a
GPU box and reaches them over the **openpi/0 protocol**: what's "remote" is the *model*,
not the agent, so the harness runs on a laptop. Serve one yourself from
`inventory/agents/remote/`, or point `--remote` at a ready-made openpi server (NVIDIA
Cosmos, a hosted pi0.5) and skip serving entirely.

## View your traces on [hud.ai](https://hud.ai)

Every rollout you run with `HUD_API_KEY` set streams to the platform automatically — no
extra wiring. When a run finishes, open [hud.ai](https://hud.ai) and click the job. The
trace viewer replays your episode end to end:

- **Camera feeds** scrubbed on a timeline (`observation/image`, wrist cam, …)
- **State & action channels** you can toggle (`robot0_eef_pos`, gripper, …)
- **Inference ticks** — when the policy predicted each action chunk and for which steps

Scrub between ticks to watch a chunk execute; click a tick to jump to the decision point.
Use this to debug failures, compare policies, and show what your agent actually did.
