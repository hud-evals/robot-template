---
name: robot-integration
description: >-
  Integrate a robot policy (VLA / imitation model) or a simulator environment
  into the HUD robot framework using this template. Use when asked to add, port,
  wrap, or benchmark a new policy/checkpoint (e.g. "integrate OpenVLA") or a new
  simulator/benchmark (e.g. "add a MetaWorld env"), or to write a contract.
---

# Robot model & environment integration

The framework splits a robot rollout into small named seams; you implement one or
two and the SDK owns the rest (serve loop, openpi/0 wire protocol, telemetry).

```
agent side                         env side
RobotAgent  (episode loop, SDK)    Environment (control channel + tasks)
  ├─ Model   how to run a policy      └─ RobotEndpoint  start/stop/reset/result/url
  └─ Adapter env<->policy spaces            └─ RobotBridge  reset/step/get_observation
```

The two sides only ever agree through **the contract** — a JSON spec of the
embodiment's observation + action spaces that the env publishes and the agent
reads back. Get the contract right and the implementation is tiny.

**Read first:**
- The docs: https://docs.hud.ai/v6/core/robots (the canonical reference).
- The SDK source (the law when docs lag): `hud.agents.robot.*` and
  `hud.environment.robot.*` — every base class has a precise docstring.
- The nearest template example to your case (table below).

| Situation | Copy |
|---|---|
| Stock LeRobot checkpoint | `inventory/agents/smolvla_libero.py` (also `pi05_libero.py`) |
| Chunked policy w/ ensembling | `inventory/agents/vla_jepa_libero.py` |
| Non-LeRobot policy / custom wiring | `inventory/agents/scaffold/my_agent.py` |
| Served / remote model | `inventory/agents/remote/smolvla/` (+ `remote/scaffold/`) |
| Sync sim env (the default) | `environment/` (the LIBERO env — the one deployed env) |
| Sim that pins the main thread (Isaac) | `inventory/envs/robolab/` (sim_process.py + `RobotEndpoint.remote`) |
| Env scaffold + minimal contract | `inventory/envs/scaffold/` |

---

## Add a model (agent side)

Subclass `RobotAgent`; in `__init__` set `self.model` and `self.adapter`. The agent
must carry **no env-specific key names** — env wiring is read from the contract at
connect time. Set `max_steps` for the episode cap.

- **Stock LeRobot checkpoint** — `LeRobotModel(policy, pre, post)` +
  `LeRobotAdapter(model_image_keys=list(policy.config.image_features))`. That's it.
- **Non-LeRobot policy** — subclass `Model`, implement
  `infer(batch) -> [N, T, A]` (keep the leading batch dim, even for N=1).
- **Custom wiring** (resize/pad, delta→abs, gripper remap) — subclass `Adapter`;
  `adapt_observation(obs, prompt) -> batch` is the one you must write,
  `adapt_action` defaults to identity.
- **Served weights** — `RemoteModel(host, port)` + `OpenPIAdapter()`. Weights stay
  on the GPU box; only a stateless chunk forward crosses the network. Works against
  any openpi server (stock openpi, Cosmos with `response_key="action"`, pi0.5).

The harness loop: `setup_robot -> adapter.bind(spaces)` (once), `select_action ->
adapt_observation -> model.ainfer -> pop chunk -> adapt_action` (per step). It
executes chunks open-loop, re-inferring only when the active chunk is spent.

## Add an environment (env side)

Write a `RobotBridge` subclass and a declarative `env.py` (copy
`inventory/envs/scaffold/`). The bridge is three methods:

- `async reset(**task_args) -> str` — build the episode, return the prompt. (The
  base zeroes scoring + pushes the first frame around your hook.)
- `def step(action)` — **synchronous**; advance one tick; accumulate
  `total_reward`; set `success` from the **goal check only** (a timeout is
  termination, not success); set `terminated = done or success`.
- `get_observation() -> (data, terminated) | None` — `data` keys are **exactly**
  the contract's observation feature names; images raw `uint8` HWC; state one 1-D
  float array in contract order.

`start`/`stop`/`url` and the recv-action→step→send-obs loop are the base's. Pick a
`SimRunner` (`InlineSimRunner` default; `ThreadSimRunner` for heavy/blocking sims;
`MainThreadSimRunner` for Isaac/Omniverse — see `inventory/envs/robolab/`).

`env.py` drives the bridge through `RobotEndpoint` (identical local or
`.remote(...)`), publishes `Capability.robot(url=..., contract=CONTRACT)` after
`start()`, and exposes one `@env.template` async generator per task family
(two yields: the prompt, then the result).

## Write the contract

A JSON doc in the capability manifest; see `inventory/envs/scaffold/example_contract.json`
and the [field reference](https://docs.hud.ai/v6/core/robots#the-contract).

Only two fields are **load-bearing**:
- `role` (`observation` / `action`) on every feature — splits the spaces.
- `type` on image observations (`rgb`/`bgr`/`gray`/`depth`) — marks a camera; the
  first observation without an image `type` becomes the state.

Everything else (`control_rate`, `dtype`, `shape`, `names`, `stats`) is descriptive
and never enforced — but **include `shape` + `names` + `control_rate`** so the trace
viewer can label and display your obs/action slices. Feature keys must match the
bridge's `get_observation` keys verbatim; `action` is the single action feature.
One contract = one embodiment, one obs space, one action space.

## Verify (don't skip)

1. Serve the env: `python -m hud.environment.server environment/env.py --port 9001`
   (or `inventory/envs/<name>/env.py` for a library env) and open the MJPEG view at
   `http://localhost:8080/`.
2. Run a few episodes against it. A **nonzero success rate** on tasks the model was
   trained for is the bar — "it runs without crashing" is not.

Most contract bugs crash nothing and just produce a bad robot. Use the live view as
your oscilloscope: drifts-then-freezes ⇒ normalization stats; wild spinning ⇒
rotation convention; mirror-image motion ⇒ wrong `frame`; gripper inverted ⇒
gripper sign/range; slow-motion/hyperactive ⇒ control-rate or delta-as-absolute.
Fix the **contract** first, then make the implementation follow it.
