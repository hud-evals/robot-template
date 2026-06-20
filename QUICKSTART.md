# Quickstart


## 1. Install (1 min)

```bash
# The SDK + robot extra. The v6 robot API isn't on PyPI yet, so install from git
# (use the plain 'hud-python[robot]' from pyproject once the release lands). Use a
# fresh env — if an older hud-python is installed, add --force-reinstall.
pip install 'hud-python[robot] @ git+https://github.com/hud-evals/hud-python.git'
pip install 'lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git'   # the policy
```

`pyproject.toml` declares the same harness deps (`pip install -e '.[vla]'` once the
robot API is on PyPI). The heavy LIBERO sim stack isn't installed locally — it lives
inside `Dockerfile.hud` and only runs in the env container.

## 2. Verify (10 sec)

```bash
python test_install.py
```

`CORE OK` means you're ready. It also reports your torch device and which optional tools
(Docker, Modal) you have — that decides which path below fits you.

## 3. Build the env image (once)

`Dockerfile.hud` is the LIBERO env build. Build it locally so the runs below can boot it
(`DockerRuntime("hud-libero-env")`):

```bash
docker build -f Dockerfile.hud -t hud-libero-env .    # first build pulls LIBERO assets (~few min)
```

## 4. Run

```bash
python run.py                 # SmolVLA × 3 LIBERO episodes -> success_rate
python run.py --batched 8     # 8 episodes off one batched GPU forward
python run.py --full          # the whole libero_spatial suite (500 episodes)
python run.py --full --suite all   # the entire 6500-episode benchmark
```

The `examples/` scripts are the same modes spelled out one file each
(`examples/local.py`, `examples/batched.py`, `examples/remote.py`). `--full` reads the
checked-in `environment/full_bench_tasks.json` (regenerate with `python environment/tasks.py`).

### No local GPU? Run it all remotely

The model is served on a GPU and the env runs in the cloud; your machine is just the
harness. Needs a Modal account (`pip install modal && modal token new`).

```bash
# 1) serve the policy on a Modal GPU — prints  ws://HOST:PORT
modal run inventory/agents/remote/smolvla/smolvla_serve.py
export POLICY_HOST=<host>  POLICY_PORT=<port>

# 2) publish the env image on Modal (once) — builds Dockerfile.hud
modal run inventory/envs/remote/modal/deploy.py

# 3) drive it from here (no GPU needed locally)
python run.py --remote "$POLICY_HOST:$POLICY_PORT" --modal
```

Already have an openpi server (NVIDIA Cosmos, a hosted pi0.5)? Skip step 1 and point
`--remote` at it.

## Deploy to the HUD platform

Instead of running the env yourself, host it on HUD and run hosted:

```bash
hud deploy                    # builds Dockerfile.hud, registers the env as "libero"
```

## API keys

| Var | When you need it |
| --- | --- |
| `HUD_API_KEY` | stream telemetry / traces to the HUD platform (recommended; get one at hud.ai) |
| Modal token (`modal token new`) | the remote model server + Modal env runtime |
| `HF_TOKEN` | gated checkpoints, and pushing recorded datasets to the Hub |
| `DAYTONA_API_KEY` | the Daytona env runtime (`inventory/envs/remote/daytona/`) |

## Where next

- **Run your own policy** → copy `inventory/agents/scaffold/my_agent.py`.
- **Add your own simulator** → copy `inventory/envs/scaffold/`, then promote it to
  `environment/` and point `Dockerfile.hud` at it.
- **Serve your model remotely** → copy `inventory/agents/remote/scaffold/`.
- **Concepts & full reference** → [https://docs.hud.ai/v6/core/robots](https://docs.hud.ai/v6/core/robots)
